import jax

from . import ctx
from .module import Module


def _deep_scan(mod):
    if isinstance(mod, Module):
        mod.deep_scan()


def enable_strict_mode(f):
    """The strict mode includes two things:
    
    - enable immutability mode.
    - call deep_scan on input modules.
    """
    def wrapper(fn, *args, **kwds):
        assert callable(fn), "Expecting a callable object as the first argument."

        def fake_fn(*u, **v):

            # scan for bugs
            jax.tree_map(_deep_scan, (u, v), is_leaf=lambda x: isinstance(x, Module))

            with ctx.immutable():
                return fn(*u, **v)

        return f(fake_fn, *args, **kwds)

    return wrapper


jit = enable_strict_mode(jax.jit)
vmap = enable_strict_mode(jax.vmap)
pmap = enable_strict_mode(jax.pmap)
grad = enable_strict_mode(jax.grad)
