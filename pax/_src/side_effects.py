import functools

import jax

from .module import Module


def keep_side_effects(f):
    @functools.wraps(f)
    def _f(fn, **v):
        has_aux = v.get("has_aux", False)

        def get_modules(v):
            modules = jax.tree_flatten(v, is_leaf=lambda x: isinstance(x, Module))[0]
            modules = [m for m in modules if isinstance(m, Module)]
            return modules

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            modules = get_modules((args, kwargs))
            from .transforms import scan_bugs

            [scan_bugs(mod) for mod in modules]
            out = fn(*args, **kwargs)
            [scan_bugs(mod) for mod in modules]
            out_modules = get_modules(out)
            [scan_bugs(mod) for mod in out_modules]
            if f in [jax.value_and_grad, jax.grad] and has_aux:
                out, aux = out
                return out, (aux, *modules)
            else:
                return out, modules

        if f in [jax.value_and_grad, jax.grad]:
            v["has_aux"] = True

        __fn = f(_fn, **v)

        @functools.wraps(__fn)
        def ___fn(*x, **y):
            modules = get_modules((x, y))
            out = __fn(*x, *y)

            if f == jax.grad and has_aux:
                out, (aux, *updated_modules) = out
                out = (out, aux)
            elif f == jax.value_and_grad and has_aux:
                (out, (aux, *updated_modules)), grads = out
                out = ((out, aux), grads)
            elif f == jax.value_and_grad and not has_aux:
                (out, updated_modules), grads = out
                out = (out, grads)
            else:
                out, updated_modules = out

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
grad_ = keep_side_effects(jax.grad)
value_and_grad_ = keep_side_effects(jax.value_and_grad)
