"""Thin wrappers for jax transformation."""
from functools import wraps
from typing import Callable, TypeVar, Union

import jax

from . import ctx

C = TypeVar("C", bound=Callable)


def _deep_scan(x):
    from .module import Module
    from .transforms import scan_bugs

    def __deep_scan(v):
        if isinstance(v, Module):
            return scan_bugs(v)
        else:
            return v

    x = jax.tree_map(__deep_scan, x, is_leaf=lambda v: isinstance(v, Module))
    return x


def _get_all_module_treedefs(v):
    from .module import Module
    from .utils import EmptyNode

    leaves = jax.tree_flatten(v, is_leaf=lambda x: isinstance(x, Module))[0]
    mods = [x for x in leaves if isinstance(x, Module)]
    tree_defs = [
        jax.tree_flatten(x, is_leaf=lambda x: isinstance(x, EmptyNode))[1] for x in mods
    ]
    return set(tree_defs)


def enable_strict_mode(f):
    """The strict mode includes safeguards:

    - call deep_scan on input & output modules.
    - enable immutable mode.
    - use copy of the inputs to prevent side effects.
    - a function must return the updated input modules.
    """

    @wraps(f)
    def wrapper(
        fn: C,
        *args,
        deep_scan: bool = True,
        copy: bool = False,
        io_check: bool = False,
        **kwargs,
    ) -> Union[Callable, C]:
        """Jax transformation with some additional arguments.

        Arguments:
            deep_scan: scan inputs and outputs for bugs. Default: True
            copy: copy inputs to avoid side effects. Default: True
            io_check: a function must returns the updated input modules. Default: False
        """
        assert callable(fn), "Expecting a callable object as the first argument."

        @wraps(fn)
        def _fn(*u, **v):

            # scan inputs for bugs
            if deep_scan:
                _deep_scan((u, v))

            # use copy of the inputs to prevent side effects
            # therefore, the function `f` has to returns modified
            # objects as its outputs.
            if copy:
                u, v = jax.tree_map(lambda x: x, (u, v))

            if io_check:
                input_treedefs = _get_all_module_treedefs((u, v))

            out = fn(*u, **v)

            # scan outputs for bugs
            if deep_scan:
                _deep_scan(out)

            if io_check:
                output_treedefs = _get_all_module_treedefs(out)
                if input_treedefs != output_treedefs:
                    raise ValueError(
                        f"In Pax's strict mode, a function must return the updated versions of input modules.\n"
                        f"\n"
                        f"Input treedefs:  {input_treedefs}\n"
                        f"Output treedef:  {output_treedefs}\n"
                    )

            return out

        return f(_fn, *args, **kwargs)

    return wrapper


jit = enable_strict_mode(jax.jit)
vmap = enable_strict_mode(jax.vmap)
pmap = enable_strict_mode(jax.pmap)
grad = enable_strict_mode(jax.grad)
