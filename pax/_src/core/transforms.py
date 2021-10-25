"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, TypeVar

import jax

from .base import BaseModule, EmptyNode, PaxKind

TreeDef = Any

T = TypeVar("T", bound=BaseModule)
K = TypeVar("K", bound=BaseModule)
O = TypeVar("O", bound=BaseModule)


def _update_pax(mod, pax_info):
    super(BaseModule, mod).__setattr__("_pax", pax_info)
    return mod


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""

    def _train_apply_fn(mod: T) -> T:
        # pylint: disable=protected-access
        return _update_pax(mod, mod._pax._replace(training=True))

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation mode."""

    def _eval_apply_fn(mod: T) -> T:
        # pylint: disable=protected-access
        return _update_pax(mod, mod._pax._replace(training=False))

    return mod.apply(_eval_apply_fn)


def freeze_parameters(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        new_name_to_kind = OrderedDict()
        # pylint: disable=protected-access
        for (name, kind) in mod._pax.name_to_kind.items():
            if kind == PaxKind.PARAMETER:
                new_name_to_kind[name] = PaxKind.STATE
            else:
                new_name_to_kind[name] = kind

        # use proxy to avoid any side effects
        # pylint: disable=protected-access
        pax_info = mod._pax._replace(name_to_kind=MappingProxyType(new_name_to_kind))
        return _update_pax(mod, pax_info)

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameters(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    assert freeze_parameters(origin) == mod
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_kind(mod: T, *, kind: PaxKind) -> T:
    """Select leaves of kind ``kind`` while setting all other leaves to ``None``.

    Arguments:
        mod: The module.
        kind: The kind of leaves that will be kept intact.
    """
    assert kind in [PaxKind.PARAMETER, PaxKind.STATE]
    if kind == PaxKind.STATE:
        none_list = [PaxKind.PARAMETER]
    else:
        none_list = [PaxKind.STATE]

    def _select_apply_fn(mod: T) -> T:
        # pylint: disable=protected-access
        for (name, kind) in mod._pax.name_to_kind.items():
            if kind in none_list:
                value = getattr(mod, name)
                none_v = jax.tree_map(lambda _: EmptyNode(), value)
                mod.__dict__[name] = none_v
        return mod

    return mod.apply(_select_apply_fn)


def select_parameters(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return select_kind(mod, kind=PaxKind.PARAMETER)


def select_states(mod: T) -> T:
    """Select `STATE` leaves only."""
    return select_kind(mod, kind=PaxKind.STATE)


def update_pytree(mod: T, *, other: T) -> T:
    """Use non-EmptyNode leaves from others"""

    def _select_fn(leaf_x, leaf_y):
        if isinstance(leaf_y, EmptyNode):
            return leaf_x
        else:
            return leaf_y

    new_mod = jax.tree_map(_select_fn, mod, other)
    new_mod = jax.tree_unflatten(jax.tree_structure(mod), jax.tree_leaves(new_mod))
    return new_mod


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module that uses trainable parameters in `params`."""
    return update_pytree(mod, other=select_parameters(params))


def update_states(mod: T, *, states: T) -> T:
    """Return a module that uses non-trainable states in `states`."""
    return update_pytree(mod, other=select_states(states))
