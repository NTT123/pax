"""Optax optimizers as Pax modules."""

import logging
from typing import List, TypeVar

import jax
import jax.tree_util
import optax

from . import tree
from .module import Module

T = TypeVar("T", bound="Module")


def from_optax(optax_obj: optax.GradientTransformation):
    """Build an optimizer from optax.

    Arguments:
        optax_obj: The optimizer object created by optax.

    Returns:
        OptaxOptimizer: A Module optimizer."""

    class OptaxOptimizer(Module):
        state: List[optax.OptState]

        def __init__(self, params: T):
            super().__init__()
            self.state = tree.StateTree(optax_obj.init(params))

        def step(self, grads: T, params: T) -> T:
            """Update model parameters and optimizer state.

            Arguments:
                grads: gradient tree.
                params: parameter tree.

            Returns:
                new_params: updated params.
            """
            if jax.tree_structure(params) != jax.tree_structure(grads):
                logging.error(
                    """parameter's structure is different from gradient's structure. 
                    This is likely due to updates of the model's data fields in the forward pass."""
                )
            updates, self.state = optax_obj.update(grads, self.state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params

    return OptaxOptimizer
