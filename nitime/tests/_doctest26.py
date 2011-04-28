"""Code taken from the Python2.6 standard library for backwards compatibility.

This is just so we can use 2.6 features when running in 2.5, the code below is
copied verbatim from the stdlib's collections and doctest modules.
"""

from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter
import sys as _sys


def namedtuple(typename, field_names, verbose=False):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', 'x y')
    >>> Point.__doc__            # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)      # instantiate with positional args or keywords
    >>> p[0] + p[1]              # indexable like a plain tuple
    33
    >>> x, y = p                 # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                # fields also accessable by name
    33
    >>> d = p._asdict()          # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)               # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)  #_replace() like str.replace(), targets named fields
    Point(x=100, y=22)

    """

    # Parse and validate the field names.  Validation serves two purposes,
    # generating informative error messages and preventing template injection
    # attacks.
    if isinstance(field_names, basestring):
        field_names = field_names.replace(',', ' ').split()  # names separated
                                                             # by whitespace
                                                             # and/or commas
    field_names = tuple(map(str, field_names))
    for name in (typename,) + field_names:
        if not all(c.isalnum() or c == '_' for c in name):
            e_s = 'Type names and field names can only contain alphanumeric'
            e_s += 'characters and underscores: %r' % name
            raise ValueError(e_s)
        if _iskeyword(name):
            e_s = 'Type names and field names cannot be a keyword: %r' % name
            raise ValueError(e_s)
        if name[0].isdigit():
            e_s = 'Type names and field names cannot'
            e_s += 'start with a number: %r' % name
            raise ValueError()
    seen_names = set()
    for name in field_names:
        if name.startswith('_'):
            e_s = 'Field names cannot start with an underscore: %r' % name
            raise ValueError(e_s)
        if name in seen_names:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen_names.add(name)

    # Create and fill-in the class template
    numfields = len(field_names)
    argtxt = repr(field_names).replace("'", "")[1:-1]  # tuple repr without
                                                        # parens or quotes
    reprtxt = ', '.join('%s=%%r' % name for name in field_names)
    dicttxt = ', '.join('%r: t[%d]' % (name, pos) for pos,
                        name in enumerate(field_names))
    template = '''class %(typename)s(tuple):
        '%(typename)s(%(argtxt)s)' \n
        __slots__ = () \n
        _fields = %(field_names)r \n
        def __new__(_cls, %(argtxt)s):
            return _tuple.__new__(_cls, (%(argtxt)s)) \n
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new %(typename)s object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != %(numfields)d:
                raise TypeError('Expected %(numfields)d arguments, got %%d' %% len(result))
            return result \n
        def __repr__(self):
            return '%(typename)s(%(reprtxt)s)' %% self \n
        def _asdict(t):
            'Return a new dict which maps field names to their values'
            return {%(dicttxt)s} \n
        def _replace(_self, **kwds):
            'Return a new %(typename)s object replacing specified fields with new values'
            result = _self._make(map(kwds.pop, %(field_names)r, _self))
            if kwds:
                raise ValueError('Got unexpected field names: %%r' %% kwds.keys())
            return result \n
        def __getnewargs__(self):
            return tuple(self) \n\n''' % locals()
    for i, name in enumerate(field_names):
        template += '        %s = _property(_itemgetter(%d))\n' % (name, i)
    if verbose:
        print template

    # Execute the template string in a temporary namespace and
    # support tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(_itemgetter=_itemgetter, __name__='namedtuple_%s' % typename,
                     _property=property, _tuple=tuple)
    try:
        exec template in namespace
    except SyntaxError, e:
        raise SyntaxError(e.message + ':\n' + template)
    result = namespace[typename]

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example).
    if hasattr(_sys, '_getframe'):
        result.__module__ = _sys._getframe(1).f_globals.get('__name__', '__main__')

    return result


TestResults = namedtuple('TestResults', 'failed attempted')
