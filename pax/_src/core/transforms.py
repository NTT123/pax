"""Transform a module to a new one."""
from typing import Any, TypeVar

import jax

from .module import Module, parameters_method, update_pytree

TreeDef = Any

T = TypeVar("T", bound=Module)
K = TypeVar("K", bound=Module)
O = TypeVar("O", bound=Module)


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""
    return mod.train()


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation mode."""
    return mod.eval()


def freeze_parameters(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        return mod.replace_method(parameters=parameters_method())

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameters(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_parameters(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return mod.parameters()


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module that uses trainable parameters in `params`."""
    return update_pytree(mod, other=params.parameters())
