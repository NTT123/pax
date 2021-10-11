"""PAX mechanisms to make PAX functions pure."""

import functools
import inspect

import jax

from .base import allow_mutation, enable_deep_copy
from .rng import get_rng_state, set_rng_state
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

    rng_state = get_rng_state()

    @functools.wraps(func)
    def _f(*args, **kwargs):
        _ = [m.scan_bugs() for m in get_modules((func, args, kwargs))]

        # support calling method
        if inspect.ismethod(func):
            self = (func.__self__,)
            unbound_func = func.__func__
        else:
            self = ()
            unbound_func = func

        with enable_deep_copy():
            leaves, treedef = jax.tree_flatten((self, unbound_func, args, kwargs))
        self, unbound_func, args, kwargs = jax.tree_unflatten(treedef, leaves)
        modules = _get_all_submodules((self, unbound_func, args, kwargs))
        with allow_mutation(modules):
            set_rng_state(rng_state)
            out = unbound_func(*self, *args, **kwargs)

        _ = [m.scan_bugs() for m in get_modules(out)]

        return out

    return _f
