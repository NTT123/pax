import functools

import jax


def _get_all_submodules(value):
    from .utils import get_modules

    submods = get_modules(value)
    out = list(submods)
    for mod in submods:
        out.extend(_get_all_submodules(mod.submodules()))
    return out


def pure(f):
    """Make a function pure by copying the inputs.

    Any modifications on the copy will not affect the original inputs.

    **Note**: only functions that wrapped by `pax.pure` are allowed to modify PAX's Modules.

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

    """
    from .ctx import allow_mutation, enable_deep_copy
    from .utils import get_modules

    @functools.wraps(f)
    def _f(*args, **kwargs):
        from .transforms import scan_bugs

        [scan_bugs(m) for m in get_modules((args, kwargs))]

        with enable_deep_copy():
            leaves, treedef = jax.tree_flatten((args, kwargs))
        args, kwargs = jax.tree_unflatten(treedef, leaves)

        with allow_mutation(_get_all_submodules((args, kwargs))):
            out = f(*args, **kwargs)

        [scan_bugs(m) for m in get_modules(out)]

        return out

    return _f
