"""Useful functions."""

from typing import Any, Callable, Tuple, TypeVar

import jax
import jax.numpy as jnp

from pax.module import Module
from pax.optim import Optimizer

T = TypeVar("T", bound="Module")

LossFn = Callable[[T, T, Any], Tuple[jnp.ndarray, Tuple[jnp.ndarray, T]]]
UpdateFn = Callable[[T, Optimizer, Any], Tuple[Any, T, Optimizer]]


def build_update_fn(loss_fn: LossFn) -> UpdateFn:
    """Build a simple update function."""

    def _update_fn(model: T, optimizer: Optimizer, inputs: Any):
        """An update function.

        Note that: ``model`` and ``optimizer`` have internal states.
        We have to return them in the output as jax transformations (e.g., ``jax.grad`` and ``jax.jit``) requires pure functions.


        Arguments:
            model: a callable tx.Module
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
