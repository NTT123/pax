"""Thin wrappers for jax transformation."""
import jax

from . import ctx
from .module import Module


def _deep_scan(mod):
    from .transforms import scan_bugs

    if isinstance(mod, Module):
        scan_bugs(mod)


def _get_all_module_treedefs(v):
    leaves = jax.tree_flatten(v, is_leaf=lambda x: isinstance(x, Module))[0]
    mods = [x for x in leaves if isinstance(x, Module)]
    mods = [x for x in mods if not (None in jax.tree_leaves(x))]
    return set([jax.tree_flatten(x, is_leaf=lambda x: x is None)[1] for x in mods])


def enable_strict_mode(f):
    """The strict mode includes four safeguards:

    - call deep_scan on input modules.
    - enable immutable mode.
    - use copy of the inputs to prevent side effects.
    - a function must return the updated input modules.
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

                input_treedefs = _get_all_module_treedefs((u, v))

                out = fn(*u, **v)

                output_treedefs = _get_all_module_treedefs(out)
                if input_treedefs != output_treedefs:
                    raise ValueError(
                        f"In Pax's strict mode, a function must return the updated versions of input modules.\n"
                        f"\n"
                        f"Input treedefs:  {input_treedefs}\n"
                        f"Output treedef:  {output_treedefs}\n"
                    )

                return out

        return f(fake_fn, *args, **kwds)

    return wrapper


jit = enable_strict_mode(jax.jit)
vmap = enable_strict_mode(jax.vmap)
pmap = enable_strict_mode(jax.pmap)
grad = enable_strict_mode(jax.grad)
