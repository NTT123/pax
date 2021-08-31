"""Useful functions."""

import inspect
from typing import Any, Callable, List, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp

from .module import Module
from .optim import Optimizer

T = TypeVar("T", bound="Module")

LossFnOutput = Tuple[jnp.ndarray, Tuple[jnp.ndarray, T]]
LossFn = Callable[[T, T, Any], LossFnOutput]
UpdateFn = Callable[[T, Optimizer, Any], Tuple[Any, T, Optimizer]]


def build_update_fn(loss_fn: LossFn) -> UpdateFn:
    """Build a simple update function."""

    sig = inspect.signature(loss_fn)
    parameters = sig.parameters
    if (
        list(parameters.keys()) != ["params", "model", "inputs"]
        or sig.return_annotation != LossFnOutput
    ):
        raise ValueError(
            """Expecting a loss function with an _exact_ signature:  
        ``(params, model, inputs) -> LossFnOutput``
        """
        )

    def _update_fn(model: T, optimizer: Optimizer, inputs: Any):
        """An update function.

        Note that: ``model`` and ``optimizer`` have internal states.
        We have to return them in the output as jax transformations (e.g., ``jax.grad`` and ``jax.jit``) requires pure functions.


        Arguments:
            model: a callable pax.Module
            optimizer: an optimizer
            inputs: input batch.

        Returns:
            loss: the loss value
            model: updated model
            optimizer: updated optimizer
        """
        grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(
            model.parameters(), model, inputs
        )
        model = optimizer.step(grads, model)
        return loss, model, optimizer

    return _update_fn


class Lambda(Module):
    """A pure functional module.


    Note: We put ``Lambda`` module definition here so both ``haiku.*`` and ``nn.*`` modules can use it.
    """

    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __repr__(self) -> str:
        return f"Fx[{self.f}]"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.f == jax.nn.relu:
            name = "relu"
        else:
            name = f"{self.f}"
        output = f"x => {name}(x)"
        return [output] if return_list else output
