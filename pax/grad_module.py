from typing import Callable, TypeVar

from .module import Module
from .strict_mode import grad

T = TypeVar("T", bound=Module)


def grad_module(fun: Callable, **v):
    """This is a wrapper of ``pax.grad``.

    It returns a gradient function which computes the gradients with respect to trainable parameters of a Pax's Module.
    """

    if "argnums" in v:
        assert v["argnums"] == 0, "Only support `argnums=0`."

    if "has_aux" in v:
        assert v["has_aux"] == True, "Only support `has_aux=True`."

    def f(params: T, mod: T, *args, **kwargs):
        mod = mod.update_parameters(params)
        output = fun(mod, *args, **kwargs)
        return output

    v["has_aux"] = True
    _grad_fn = grad(f, **v)

    def _grad_module_fn(mod: T, *args, **kwargs):
        if not isinstance(mod, Module):
            raise ValueError("Expecting a Pax's Module at the first argument.")
        params = mod.parameters()
        return _grad_fn(params, mod, *args, **kwargs)

    return _grad_module_fn
