"""Transform a module to a new one."""
from typing import Any, TypeVar

import jax

from .base import BaseModule, EmptyNode, parameters_method

TreeDef = Any

T = TypeVar("T", bound=BaseModule)
K = TypeVar("K", bound=BaseModule)
O = TypeVar("O", bound=BaseModule)


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""

    def _train_apply_fn(mod: T) -> T:
        # pylint: disable=protected-access
        return mod.replace(_training=True)

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation mode."""

    def _eval_apply_fn(mod: T) -> T:
        # pylint: disable=protected-access
        return mod.replace(_training=False)

    return mod.apply(_eval_apply_fn)


def freeze_parameters(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        return mod.replace_method(parameters=parameters_method([]))

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameters(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_parameters(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return mod.parameters()


def update_pytree(mod: T, *, other: T) -> T:
    """Use non-EmptyNode leaves from other."""

    def _select_fn(leaf_x, leaf_y):
        if isinstance(leaf_y, EmptyNode):
            return leaf_x
        else:
            return leaf_y

    is_empty = lambda x: isinstance(x, EmptyNode)
    new_mod = jax.tree_map(_select_fn, mod, other, is_leaf=is_empty)
    return new_mod


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module that uses trainable parameters in `params`."""
    return update_pytree(mod, other=params.parameters())
