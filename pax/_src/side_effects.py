import jax
from .module import Module
import functools


def keep_side_effects(f):
    @functools.wraps(f)
    def _f(fn, *u, **v):
        def get_modules(v):
            modules = jax.tree_flatten(v, is_leaf=lambda x: isinstance(x, Module))[0]
            modules = [m for m in modules if isinstance(m, Module)]
            return modules

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            modules = get_modules((args, kwargs))
            out = fn(*args, **kwargs)
            return out, modules

        __fn = f(_fn, *u, **v)

        @functools.wraps(__fn)
        def ___fn(*x, **y):
            modules = get_modules((x, y))
            out, updated_modules = __fn(*x, *y)
            assert len(modules) == len(updated_modules)
            for m, um in zip(modules, updated_modules):
                assert type(m) == type(um)
                m.__dict__.update(um.__dict__)

            return out

        return ___fn

    return _f


jit_ = keep_side_effects(jax.jit)
vmap_ = keep_side_effects(jax.vmap)
pmap_ = keep_side_effects(jax.pmap)
