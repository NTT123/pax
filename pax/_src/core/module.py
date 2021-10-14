"""PAX module."""

from typing import Any, Optional, TypeVar

import jax
import jax.tree_util

from .base import BaseModule, PaxFieldKind
from .threading_local import allow_mutation
from .transforms import (
    enable_eval_mode,
    enable_train_mode,
    select_parameters,
    update_parameters,
)

T = TypeVar("T", bound="Module")
M = TypeVar("M")
TreeDef = Any


class Module(BaseModule):
    """The Module class."""

    @property
    def training(self) -> bool:
        """If a module is in training mode."""
        return self._pax.training

    @property
    def name(self) -> Optional[str]:
        """Return the name of the module."""
        return self._pax.name

    def register_parameter(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER`` in the ``name_to_kind`` dictionary."""
        self.register_subtree(name, value, PaxFieldKind.PARAMETER)

    def register_state(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE`` in the ``name_to_kind`` dictionary."""
        self.register_subtree(name, value, PaxFieldKind.STATE)

    def register_modules(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE`` in the ``name_to_kind`` dictionary."""
        self.register_subtree(name, value, PaxFieldKind.MODULE)

    register_parameters = register_parameter
    register_states = register_state
    register_module = register_modules

    def copy(self: T) -> T:
        """Return a copy of the current module."""
        leaves, treedef = jax.tree_flatten(self)
        return jax.tree_unflatten(treedef, leaves)

    def train(self: T) -> T:
        """Return a module in training mode."""
        return enable_train_mode(self)

    def eval(self: T) -> T:
        """Return a module in evaluation mode."""
        return enable_eval_mode(self)

    def parameters(self: T) -> T:
        """Return trainable parameters."""
        return select_parameters(self)

    def update_parameters(self: T, params: T) -> T:
        """Return a new module with updated parameters."""
        return update_parameters(self, params=params)

    def replace(self: T, **kwargs) -> T:
        """Return a new module with some attributes replaced."""

        mod = self.copy()
        with allow_mutation(mod):
            for name, value in kwargs.items():
                assert hasattr(mod, name)
                setattr(mod, name, value)
            mod.find_and_register_submodules()

        mod.scan_bugs()
        return mod

    def scan_bugs(self: T) -> T:
        """Scan the module for potential bugs."""

        def _scan_apply_fn(mod: T) -> T:
            assert isinstance(mod, Module)
            mod._scan_fields(mod.__class__.__dict__)
            mod._scan_fields(mod.__dict__)
            return mod

        self.apply(_scan_apply_fn)
        return self
