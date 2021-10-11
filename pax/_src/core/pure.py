import functools
import inspect

import jax

from .base import allow_mutation, enable_deep_copy
from .utils import get_modules


def _get_all_submodules(value):
    submods = get_modules(value)
    out = list(submods)
    for mod in submods:
        out.extend(_get_all_submodules(mod.submodules()))
    return out


def pure(func):
    """Make a function pure by copying the inputs.

    Any modification on the copy will not affect the original inputs.

    **Note**: only functions that wrapped by `pax.pure` are allowed to modify PAX's Modules.

    Example:

    >>> f = pax.nn.Linear(3,3)
    >>> f.a_list = []
    [...]
    ValueError: Cannot modify a module in immutable mode.
    Please do this computation inside a @pax.pure function.
    >>>
    >>> @pax.pure
    ... def add_list(m):
    ...     m.a_list = []
    ...     return m
    ...
    >>> f = add_list(f)
    >>> print(f.a_list)
    []
    """

    @functools.wraps(func)
    def _f(*args, **kwargs):
        [m.scan_bugs() for m in get_modules((func, args, kwargs))]

        # support calling method
        if inspect.ismethod(func):
            self = (func.__self__,)
            fn = func.__func__
        else:
            self = ()
            fn = func

        with enable_deep_copy():
            leaves, treedef = jax.tree_flatten((self, fn, args, kwargs))
        self, fn, args, kwargs = jax.tree_unflatten(treedef, leaves)
        modules = _get_all_submodules((self, fn, args, kwargs))
        with allow_mutation(modules):
            out = fn(*self, *args, **kwargs)

        [m.scan_bugs() for m in get_modules(out)]

        return out

    return _f
