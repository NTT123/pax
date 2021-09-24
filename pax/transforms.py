"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any

import jax

TreeDef = Any
from .module import Module, PaxFieldKind, T


def enable_train_mode(mod: T) -> T:
    def _train_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = True
        return mod

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    def _eval_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = False
        return mod

    return mod.apply(_eval_apply_fn)


def freeze_parameter(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        new_name_to_kind = OrderedDict()
        for k, v in mod._name_to_kind.items():
            if v == PaxFieldKind.PARAMETER:
                new_name_to_kind[k] = PaxFieldKind.STATE
            elif v == PaxFieldKind.PARAMETER_SUBTREE:
                new_name_to_kind[k] = PaxFieldKind.STATE_SUBTREE
            else:
                new_name_to_kind[k] = v

        # use proxy to avoid any side effects
        mod.__dict__["_name_to_kind"] = MappingProxyType(new_name_to_kind)
        return mod

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameter(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    assert freeze_parameter(origin) == mod
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_kind(mod: T, *, kind: PaxFieldKind) -> T:
    assert kind in [PaxFieldKind.PARAMETER, PaxFieldKind.STATE]
    if kind == PaxFieldKind.STATE:
        none_list = [PaxFieldKind.PARAMETER, PaxFieldKind.PARAMETER_SUBTREE]
    else:
        none_list = [PaxFieldKind.STATE, PaxFieldKind.STATE_SUBTREE]

    def _select_apply_fn(mod: T) -> T:
        for k, v in mod._name_to_kind.items():
            if v in none_list:
                value = getattr(mod, k)
                none_v = jax.tree_map(lambda _: None, value)
                setattr(mod, k, none_v)
        return mod

    return mod.apply(_select_apply_fn)


def select_parameter(mod: T) -> T:
    return select_kind(mod, kind=PaxFieldKind.PARAMETER)


def select_state(mod: T) -> T:
    return select_kind(mod, kind=PaxFieldKind.STATE)


def scan_bug(mod: T) -> T:
    def _scan_apply_fn(mod: T) -> T:
        assert isinstance(mod, Module)
        mod._scan_fields(mod.__class__.__dict__)
        mod._scan_fields(mod.__dict__)
        return mod

    return mod.apply(_scan_apply_fn)
