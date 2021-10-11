"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, TypeVar

import jax

from .base import BaseModule, EmptyNode, PaxFieldKind

TreeDef = Any

T = TypeVar("T", bound=BaseModule)
K = TypeVar("K", bound=BaseModule)
O = TypeVar("O", bound=BaseModule)


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""

    def _train_apply_fn(mod: T) -> T:
        mod.__dict__["_pax"] = mod._pax._replace(training=True)
        return mod

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation mode."""

    def _eval_apply_fn(mod: T) -> T:
        mod.__dict__["_pax"] = mod._pax._replace(training=False)
        return mod

    return mod.apply(_eval_apply_fn)


def freeze_parameters(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        new_name_to_kind = OrderedDict()
        for k, v in mod._pax.name_to_kind.items():
            if v == PaxFieldKind.PARAMETER:
                new_name_to_kind[k] = PaxFieldKind.STATE
            else:
                new_name_to_kind[k] = v

        # use proxy to avoid any side effects
        mod.__dict__["_pax"] = mod._pax._replace(
            name_to_kind=MappingProxyType(new_name_to_kind)
        )
        return mod

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameters(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    assert freeze_parameters(origin) == mod
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_kind(mod: T, *, kind: PaxFieldKind) -> T:
    """Select leaves of kind ``kind`` while setting all other leaves to ``None``.

    Arguments:
        mod: The module.
        kind: The kind of leaves that will be kept intact.
    """
    assert kind in [PaxFieldKind.PARAMETER, PaxFieldKind.STATE]
    if kind == PaxFieldKind.STATE:
        none_list = [PaxFieldKind.PARAMETER]
    else:
        none_list = [PaxFieldKind.STATE]

    def _select_apply_fn(mod: T) -> T:
        for k, v in mod._pax.name_to_kind.items():
            if v in none_list:
                value = getattr(mod, k)
                none_v = jax.tree_map(lambda _: EmptyNode(), value)
                mod.__dict__[k] = none_v
        return mod

    return mod.apply(_select_apply_fn)


def select_parameters(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.PARAMETER)


def select_states(mod: T) -> T:
    """Select `STATE` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.STATE)


def update_pytree(mod: T, *, other: T) -> T:
    """Use non-EmptyNode leaves from others"""

    def _select_fn(x, y):
        if isinstance(y, EmptyNode):
            return x
        else:
            return y

    new_mod = jax.tree_map(_select_fn, mod, other)
    new_mod = jax.tree_unflatten(jax.tree_structure(mod), jax.tree_leaves(new_mod))
    return new_mod


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module which uses trainable parameters in `params`."""
    return update_pytree(mod, other=select_parameters(params))


def update_states(mod: T, *, states: T) -> T:
    """Return a module which uses non-trainable states in `states`."""
    return update_pytree(mod, other=select_states(states))
