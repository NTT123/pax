"""PAX mechanisms to make PAX method pure."""

from functools import partial
from types import MethodType
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

from .module import Module
from .pure import pure

O = TypeVar("O")
T = TypeVar("T", bound=Module)


def module_and_value(
    module_or_method: Callable[..., O],
    static_argnums: Optional[Union[int, Sequence[int]]] = None,
    check_leaks: bool = True,
) -> Callable[..., Tuple[T, O]]:
    """Return a pure function that executes a module's method.

    This pure function also returns the updated input module in the output.

    Example:

    >>> net = pax.nn.Linear(1, 1)
    >>> x = jnp.ones((32, 1))
    >>> net, y = pax.module_and_value(net)(x)  # note: `net` is also returned.


    Arguments:
        module_or_method: Either a PAX module or a method of a PAX module.
        static_argnums: a list of static arguments.
        check_leaks: enable jax leak checking.

    Returns:
        A pure function.
    """
    if isinstance(module_or_method, MethodType):  # a method
        mod = module_or_method.__self__
        func = module_or_method.__func__
    elif isinstance(module_or_method, Module):  # a module
        mod = module_or_method
        assert hasattr(mod, "__call__"), "Expecting a callable module."
        func = module_or_method.__call__.__func__
    else:
        raise ValueError("Expecting a module or a module's method.")

    assert isinstance(mod, Module), "Expecting a PAX module."

    if static_argnums is not None:
        if isinstance(static_argnums, int):
            static_argnums = [static_argnums]
        # offset by 1 for `self` argument.
        static_argnums = tuple(x + 1 for x in static_argnums)

    @partial(pure, static_argnums=static_argnums, check_leaks=check_leaks)
    def _run(mod, *args, **kwargs):
        out = func(mod, *args, **kwargs)
        return mod, out

    return partial(_run, mod)
