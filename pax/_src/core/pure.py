"""PAX mechanisms to make PAX functions pure."""

import functools
import gc
from types import MethodType
from typing import Callable, Optional, Sequence, Union

import jax

from .base import BaseModule
from .threading_local import (
    allow_mutation,
    enable_deep_copy,
    get_rng_state,
    set_rng_state,
)
from .utils import get_modules


def _get_all_submodules(value):
    submods = get_modules(value)
    out = list(submods)
    for mod in submods:
        out.extend(_get_all_submodules(mod.submodules()))
    return out


def pure(
    func: Callable,
    static_argnums: Optional[Union[int, Sequence[int]]] = None,
    check_leaks: bool = True,
) -> Callable:
    """Make a function pure by copying the inputs.

    Any modification on the copy will not affect the original inputs.

    **Note**: only functions that are wrapped by `pax.pure` are allowed to modify PAX's Modules.

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

    Arguments:
        func: A function.
        static_argnums: a list of static arguments.
        check_leaks: enable jax leak checking.

    Returns:
        A pure function.
    """

    rng_state = get_rng_state()

    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)

    if static_argnums is None:
        static_argnums = ()

    def _deepcopy(value):
        with enable_deep_copy():
            leaves, treedef = jax.tree_flatten(value)
        return jax.tree_unflatten(treedef, leaves)

    @functools.wraps(func)
    def _f(*args, **kwargs):
        _ = [m.scan_bugs() for m in get_modules((func, args, kwargs))]

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

        args = list(args)
        args_copy = tuple(args)
        for i in static_argnums:
            args[i] = None
        args = tuple(args)

        def no_leak_func(*args, **kwargs):
            gc.collect()
            args = list(args)
            for i in static_argnums:
                args[i] = args_copy[i]
            args = tuple(args)
            set_rng_state(rng_state)
            out = unbound_func(*args, **kwargs)
            set_rng_state(rng_state)
            gc.collect()
            return out

        def _run(args, kwargs, eval_shape: bool = False):
            args, kwargs = _deepcopy((args, kwargs))
            modules = _get_all_submodules((args, kwargs))
            with allow_mutation(modules):
                if eval_shape:
                    out = jax.eval_shape(no_leak_func, *args, **kwargs)
                else:
                    out = no_leak_func(*args, **kwargs)
                return out

        with jax.check_tracer_leaks(check_leaks):
            # leak check
            _run(args, kwargs, eval_shape=True)
            # real run
            out = _run(args, kwargs, eval_shape=False)
        _ = [m.scan_bugs() for m in get_modules(out)]
        return out

    return _f
