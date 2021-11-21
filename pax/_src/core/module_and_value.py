"""PAX mechanisms to make PAX method pure."""

from functools import partial
from types import MethodType
from typing import Callable, Tuple, TypeVar

from .base import BaseModule
from .pure import pure

O = TypeVar("O")
T = TypeVar("T", bound=BaseModule)


def module_and_value(module_or_method: Callable[..., O]) -> Callable[..., Tuple[T, O]]:
    """Return a pure function that executes a module's method.

    This pure function also returns the updated input module in the output.

    Example:

    >>> net = pax.Linear(1, 1)
    >>> x = jnp.ones((32, 1))
    >>> net, y = pax.module_and_value(net)(x)  # note: `net` is also returned.


    Arguments:
        module_or_method: Either a PAX module or a method of a PAX module.

    Returns:
        A pure function.
    """
    is_bound_method = True
    if isinstance(module_or_method, MethodType):  # a method
        mod = module_or_method.__self__
        func = module_or_method.__func__
    elif isinstance(module_or_method, BaseModule):  # a module
        mod = module_or_method
        assert hasattr(mod, "__call__"), "Expecting a callable module."
        func = module_or_method.__call__.__func__
    elif callable(module_or_method):
        is_bound_method = False
        func = module_or_method
    else:
        raise ValueError("Expecting a module or a module's method.")

    @pure
    def _run(mod, *args, **kwargs):
        assert isinstance(mod, BaseModule), "Expecting a PAX module."
        out = func(mod, *args, **kwargs)
        return mod, out

    if is_bound_method:
        return partial(_run, mod)
    else:
        return _run
