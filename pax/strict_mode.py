"""Thin wrappers for jax transformation."""
import jax

from . import ctx
from .module import Module


def _deep_scan(mod):
    from .transforms import scan_bugs

    if isinstance(mod, Module):
        scan_bugs(mod)


def enable_strict_mode(f):
    """The strict mode includes three things:

    - call deep_scan on input modules.
    - enable immutable mode.
    - use copy of the inputs to prevent side effects.
    """

    def wrapper(fn, *args, **kwds):
        assert callable(fn), "Expecting a callable object as the first argument."

        def fake_fn(*u, **v):

            # scan for bugs
            jax.tree_map(_deep_scan, (u, v), is_leaf=lambda x: isinstance(x, Module))

            # enable immutable mode
            with ctx.immutable():
                # use copy of the inputs to prevent side effects
                # therefore, the function `f` has to returns modified
                # objects as its outputs.
                leaves, treedef = jax.tree_flatten((u, v))
                u, v = jax.tree_unflatten(treedef=treedef, leaves=leaves)
                return fn(*u, **v)

        return f(fake_fn, *args, **kwds)

    return wrapper


jit = enable_strict_mode(jax.jit)
vmap = enable_strict_mode(jax.vmap)
pmap = enable_strict_mode(jax.pmap)
grad = enable_strict_mode(jax.grad)
