# -*- coding: utf-8 -*-

"""
Since we don't force some packages to be build-time requirements, these
functions are used to decorate methods that require said packages. Internally,
it checks on the availability of the package before calling the function. All
library imports should take place inside the method body.
"""

import functools
import warnings
import six
import time

from . import _config as cfg


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def _lazy_import_validator(delegate, *packages):
    for pkg in packages:
        if isinstance(pkg, six.string_types):
            installed = cfg._conf._see_if_available(pkg)
        elif isinstance(pkg, dict):
            if len(pkg) > 1:
                raise ValueError("Cannot interpret {0}".format(pkg))
            import_name, pkg = list(pkg.items())[0]
            installed = cfg._conf._see_if_available(import_name)

        else:
            raise TypeError("Cannot interpret {0} (type={1}). "
                            "Elements should be str or dict."
                            .format(pkg, type(pkg)))

        if not installed:
            raise ImportError("{0} requires {1}"
                              .format(delegate.__name__, pkg))


def depends_on(*packages):
    """A class or function decorator for callables that depend on packages

    Either a number of package names or mappings from package import names to
    package install names::

        >>> @depends_on('scipy', {'sklearn': 'scikit-learn'}, 'a_fake_pkg')
        ... class SomeClass:
        ...     pass
        >>> SomeClass()
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ImportError: SomeClass requires a_fake_pkg

        >>> @depends_on('scipy', 'numpy')
        ... def do_something_cool():
        ...     print('ayyy')
        >>> do_something_cool()
        ayyy
    """
    def callable_wrapper(delegate):
        orig_init = None
        if hasattr(delegate, "__init__"):  # <~~ it's a class
            # Make copy of original __init__, so we can call it without
            # recursion, but only if cls proves to be a class and not a
            # function
            orig_init = delegate.__init__

        # Returned for classes
        @functools.wraps(delegate)
        def class_init_wrapper(self, *args, **kwargs):
            _lazy_import_validator(delegate, *packages)
            orig_init(self, *args, **kwargs)  # calls the original __init__

        # Returned for functions
        @functools.wraps(delegate)
        def func_wrapper(*args, **kwargs):
            _lazy_import_validator(delegate, *packages)
            return delegate(*args, **kwargs)

        # overload the class's __init__ func if it's a class, or just return
        # the func wrapper otherwise
        if orig_init:
            delegate.__init__ = class_init_wrapper
            return delegate
        else:
            return func_wrapper
    return callable_wrapper


def deprecated(use_instead_msg):
    def func_wrapper(func):
        """A decorator that will raise deprecation warnings"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn("'{0}' is deprecated and will be removed in a "
                          "future release. Use {1} instead"
                          .format(func.__name__, use_instead_msg))
            return func(*args, **kwargs)
        return wrapper
    return func_wrapper
