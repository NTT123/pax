"""PAX mechanisms to make PAX functions pure."""

import functools
from types import MethodType
from typing import Callable

import jax

from .base import BaseModule
from .threading_local import allow_mutation, enable_deep_copy


def pure(func: Callable):
    """Make a function pure by copying the inputs.

    Any modification on the copy will not affect the original inputs.

    **Note**: only functions that are wrapped by `pax.pure` are allowed to modify PAX's Modules.

    Example:

    >>> f = pax.nn.Linear(3,3)
    >>> f.a_list = []
    Traceback (most recent call last):
      ...
    ValueError: Cannot modify a module in immutable mode.
    Please do this computation inside a function decorated by `pax.pure`.
    >>>
    >>> @pax.pure
    ... def add_list(m):
    ...     m.a_list = []
    ...     return m
    ...
    >>> f = add_list(f)
    >>> print(f.a_list)
    []

    Arguments:
        func: A function.

    Returns:
        A pure function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for m in _get_modules((func, args, kwargs)):
            m.scan_bugs()

        # support calling method
        if isinstance(func, MethodType):
            args = (func.__self__, *args)
            unbound_func = func.__func__
        # or calling a module
        elif isinstance(func, BaseModule) and callable(func):
            args = (func, *args)
            unbound_func = func.__call__.__func__
        elif callable(func):
            unbound_func = func
        else:
            raise ValueError("Not supported")

        args, kwargs = _deepcopy((args, kwargs))
        modules = _get_all_submodules((args, kwargs))
        with allow_mutation(modules):
            out = unbound_func(*args, **kwargs)

        for m in _get_modules(out):
            m.scan_bugs()
        return out

    return wrapper


def _get_modules(tree):
    "Return a list of modules in the pytree `tree`."
    modules = jax.tree_flatten(tree, is_leaf=lambda x: isinstance(x, BaseModule))[0]
    modules = [m for m in modules if isinstance(m, BaseModule)]
    return modules


def _get_all_submodules(value):
    submods = _get_modules(value)
    out = list(submods)
    for mod in submods:
        out.extend(_get_all_submodules(mod.submodules()))
    return out


def _deepcopy(value):
    with enable_deep_copy():
        leaves, treedef = jax.tree_flatten(value)
    return jax.tree_unflatten(treedef, leaves)
