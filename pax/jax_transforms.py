import jax

from . import ctx


def enable_immutable_mode(f):
    def wrapper(*args, **kwds):
        real_fn = f(*args, **kwds)

        def fake_fn(*u, **v):
            with ctx.immutable():
                return real_fn(*u, **v)

        return fake_fn

    return wrapper


jit = enable_immutable_mode(jax.jit)
vmap = enable_immutable_mode(jax.vmap)
pmap = enable_immutable_mode(jax.pmap)
grad = enable_immutable_mode(jax.grad)
