from functools import wraps

import jax

from . import ctx


def enable_immutable_mode(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        with ctx.immutable():
            return f(*args, **kwds)

    return wrapper


jit = enable_immutable_mode(jax.jit)
vmap = enable_immutable_mode(jax.vmap)
pmap = enable_immutable_mode(jax.pmap)
grad = enable_immutable_mode(jax.grad)
