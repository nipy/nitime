
from nitime.analysis.base import BaseAnalyzer
import numpy.testing as npt


def test_base():
    """Testing BaseAnalyzer"""

    empty_dict = {}
    input1 = '123'
    A = BaseAnalyzer(input=input1)

    npt.assert_equal(A.input, input1)
    npt.assert_equal(A.parameters, empty_dict)

    input2 = '456'
    A.set_input(input2)

    npt.assert_equal(A.input, input2)

    npt.assert_equal(A.__repr__(), 'BaseAnalyzer()')
