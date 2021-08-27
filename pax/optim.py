"""Optax optimizers as trex modules."""

import logging
from abc import abstractmethod
from typing import List, TypeVar, cast

import jax
import jax.tree_util
import optax

from pax.module import Module
from pax.tree import State

T = TypeVar("T", bound="Module")

_OptaxState = List[State]
# TODO: remove cast trick.
OptaxState = cast(List[optax.OptState], _OptaxState)


class Optimizer(Module):
    state: OptaxState

    @abstractmethod
    def __init__(self, params: T, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, grads: T, model: T) -> T:
        pass


def from_optax(optax_obj: optax.GradientTransformation):
    """Build an optimizer from optax.

    Arguments:
        optax_obj: The optimizer object created by optax.

    Returns:
        OptaxOptimizer: A Module optimizer."""

    class OptaxOptimizer(Optimizer):
        state: OptaxState

        def __init__(self, params: T):
            self.state = optax_obj.init(params)

        def step(self, grads: T, model: T) -> T:
            """Update model parameters and optimizer state.

            Arguments:
                grads: gradient tree.
                params: parameter tree.

            Returns:
                new_model: updated model.
            """
            params = model.parameters()
            if jax.tree_structure(params) != jax.tree_structure(grads):
                logging.error(
                    """Parameter's structure is different from gradient's structure. 
                    This is likely due to updates of the model's data fields 
                    (which are not `tx.State` or `tx.Parameter`) in the forward pass."""
                )
            updates, self.state = optax_obj.update(grads, self.state, params)
            new_params = optax.apply_updates(params, updates)
            new_model = model.update(new_params)
            return new_model

    return OptaxOptimizer


def adamw(
    params: T, learning_rate: float = 1e-4, weight_decay: float = 1e-4
) -> Optimizer:
    return from_optax(
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )(params)
