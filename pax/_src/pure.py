import functools
import inspect

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

    Any modification on the copy will not affect the original inputs.

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

        [scan_bugs(m) for m in get_modules((f, args, kwargs))]

        # support calling method
        if inspect.ismethod(f):
            self = (f.__self__,)
            fn = f.__func__
        else:
            self = ()
            fn = f

        with enable_deep_copy():
            leaves, treedef = jax.tree_flatten((self, fn, args, kwargs))
        self, fn, args, kwargs = jax.tree_unflatten(treedef, leaves)
        modules = _get_all_submodules((self, fn, args, kwargs))
        with allow_mutation(modules):
            out = fn(*self, *args, **kwargs)

        [scan_bugs(m) for m in get_modules(out)]

        return out

    return _f
