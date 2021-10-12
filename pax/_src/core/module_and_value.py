"""PAX mechanisms to make PAX method pure."""

from functools import partial
from types import MethodType
from typing import Callable, Union

from .module import Module
from .pure import pure


def module_and_value(module_or_method: Union[Module, Callable]):
    """Return a pure function that executes a module's method.

    This pure function also returns the updated input module in the output.

    Example:

    >>> net = pax.nn.Linear(1, 1)
    >>> x = jnp.ones((32, 1))
    >>> net, y = pax.module_and_value(net)(x)  # note: `net` is also returned.


    Arguments:
        module_or_method: Either a PAX module or a method of a PAX module.

    Returns:
        a pure function.
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

    @pure
    def _run(mod, *args, **kwargs):
        out = func(mod, *args, **kwargs)
        return mod, out

    return partial(_run, mod)
