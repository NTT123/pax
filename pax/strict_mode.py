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
    any_none = lambda xs: any(x is None for x in xs)
    mods = [x for x in mods if not any_none(jax.tree_leaves(x))]
    return set([jax.tree_flatten(x, is_leaf=lambda x: x is None)[1] for x in mods])


def enable_strict_mode(f):
    """The strict mode includes four safeguards:

    - call deep_scan on input modules.
    - enable immutable mode.
    - use copy of the inputs to prevent side effects.
    - a function must return the updated input modules.
    """

    def wrapper(
        fn,
        *args,
        deep_scan: bool = True,
        copy: bool = True,
        io_check: bool = True,
        **kwargs,
    ):
        """Jax transformation with some additional arguments.

        Arguments:
            deep_scan: scan inputs for bugs.
            copy: copy inputs to avoid side effects.
            io_check: an function must returns the updated input modules.
        """
        assert callable(fn), "Expecting a callable object as the first argument."

        def fake_fn(*u, **v):

            # scan for bugs
            if deep_scan:
                jax.tree_map(
                    _deep_scan, (u, v), is_leaf=lambda x: isinstance(x, Module)
                )

            # enable immutable mode
            with ctx.immutable():
                # use copy of the inputs to prevent side effects
                # therefore, the function `f` has to returns modified
                # objects as its outputs.
                if copy:
                    u, v = jax.tree_map(lambda x: x, (u, v))

                if io_check:
                    input_treedefs = _get_all_module_treedefs((u, v))

                out = fn(*u, **v)

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

        return f(fake_fn, *args, **kwargs)

    return wrapper


jit = enable_strict_mode(jax.jit)
vmap = enable_strict_mode(jax.vmap)
pmap = enable_strict_mode(jax.pmap)
grad = enable_strict_mode(jax.grad)
