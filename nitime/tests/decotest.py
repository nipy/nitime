"""Experimental code for cleaner parametric and standalone tests.

- An @as_unittest decorator can be used to tag any normal parameter-less
  function as a unittest TestCase.  Then, both nose and normal unittest will
  recognize it as such.

- A @parametric decorator tags any generator (a function with 'yield') as a
  parametric test.  This generator will get iterated to completion with each
  successful 'yield' counting as a passed test.  If there is a failure, it can
  be debugged with the function's real stack intact, instead of how nose
  approaches this problem (which leaves you inside of the nose stack without
  any access to your original stack).

  Note: this decorator still has one important limitation.  In a parametric
  test, if there is a failure, the whole run for that test stops immediately.
  This is because Python doesn't allow a generator that throws an exception to
  be resumed.  So your total test count will vary, as an early failure means
  that the rest of those sub-tests don't get run.  Each failure has to be fixed
  so the generator can continue.

Authors
-------

- Fernando Perez <Fernando.Perez@berkeley.edu>
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2009  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import unittest

from compiler.consts import CO_GENERATOR

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

# Simple example of the basic idea
def as_unittest(func):
    """Decorator to make a simple function into a normal test via unittest."""
    class Tester(unittest.TestCase):
        def test(self):
            func()

    Tester.__name__ = func.func_name

    return Tester


def isgenerator(func):
    try:
        return func.func_code.co_flags & CO_GENERATOR != 0
    except AttributeError:
        return False


class ParametricTestCase(unittest.TestCase):
    """Write parametric tests in normal unittest testcase form.

    Limitations: the last iteration misses printing out a newline when running
    in verbose mode.
    """
    def run_parametric(self, result, testMethod):
        """This replaces the run() method for test generators.

        We iterate manually the test generator."""
        testgen = testMethod()
        # We track whether the iteration finishes normally so we can adjust the
        # number of tests run at the end (in the finally: clause).  If we don't
        # do this, we get one more test reported as run than there really were.
        finished_iteration = False
        while True:
            try:
                # Initialize test
                result.startTest(self)

                # SetUp
                try:
                    self.setUp()
                except KeyboardInterrupt:
                    raise
                except:
                    result.addError(self, self._exc_info())
                    return
                # Test execution
                ok = False
                try:
                    testgen.next()
                    ok = True
                except StopIteration:
                    # We stop the loop and note that we finished normally
                    finished_iteration = True
                    break
                except self.failureException:
                    result.addFailure(self, self._exc_info())
                except KeyboardInterrupt:
                    raise
                except:
                    result.addError(self, self._exc_info())
                # TearDown
                try:
                    self.tearDown()
                except KeyboardInterrupt:
                    raise
                except:
                    result.addError(self, self._exc_info())
                    ok = False
                if ok:
                    result.addSuccess(self)
                
            finally:
                result.stopTest(self)
                # Since the startTest() method must be called (above) before we
                # can see the StopIteration exception, the last attempt bumps
                # the test count by one more than there really were tests.  In
                # this case, we must adjust the number down by one.
                if finished_iteration:
                    result.testsRun -= 1
                    
    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        testMethod = getattr(self, self._testMethodName)
        
        # Depending on the type of test method, either:
        if isgenerator(testMethod):
            # For generators, we manually iterate test generators
            return self.run_parametric(result, testMethod)
        else:
            # Or for normal tests, let the default from unittest work
            return super(ParametricTestCase, self).run(result)


def parametric(func):
    """Decorator to make a simple function into a normal test via unittest."""
    class Tester(ParametricTestCase):

        # We need a normal method that calls the original function.  Do NOT
        # fall into the temptation of using staticmethod here, because a
        # staticmethod doesn't have the proper structure of a real instance
        # method, which the rest of nose/unittest will need later.  So a simple
        # pass-through test() method that just calls func() is sufficient.
        # This method must pump the iterator from the original function, since
        # it must also be a generator.
        def test(self):
            for t in func():
                yield t

    Tester.__name__ = func.func_name

    return Tester


def count_failures(runner):
    """Count number of failures in a doctest runner.

    Code modeled after the summarize() method in doctest.
    """
    try:
        from doctest import TestResults
    except:
        from _doctest26 import TestResults

    return [TestResults(f, t) for f, t in runner._name2ft.values() if f > 0 ]
