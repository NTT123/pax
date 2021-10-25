"""PAX module."""

from typing import Any, Optional, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

from .base import BaseModule, PaxKind
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
        """Register a parameter."""
        self.register_subtree(name, value, PaxKind.PARAMETER)

    def register_state(self, name: str, value: Any):
        """Register a state."""
        self.register_subtree(name, value, PaxKind.STATE)

    def register_modules(self, name: str, value: Any):
        """Register a module subtree."""
        self.register_subtree(name, value, PaxKind.MODULE)

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

    # inspired by patrick-kidger/equinox `tree_at`
    def replace_node(self: T, node: jnp.ndarray, value: jnp.ndarray) -> T:
        """Replace a node of the pytree by a new value.

        Example:

        >>> mod = pax.nn.Sequential(
        ...     pax.nn.Linear(2,2),
        ...     jax.nn.relu
        ... )
        >>> mod = mod.replace_node(mod[0].weight, jnp.zeros((2, 3)))
        >>> print(mod[0].weight.shape)
        (2, 3)
        """
        leaves, tree_def = jax.tree_flatten(self, is_leaf=lambda x: x is node)
        count = sum(1 if x is node else 0 for x in leaves)

        if count != 1:
            raise ValueError(f"The node `{node}` appears {count} times in the module.")

        # replace `node` by value
        new_leaves = [value if v is node else v for v in leaves]
        mod: T = jax.tree_unflatten(tree_def, new_leaves)
        mod.scan_bugs()
        return mod

    def scan_bugs(self: T) -> T:
        """Scan the module for potential bugs."""

        def _scan_apply_fn(mod: T) -> T:
            assert isinstance(mod, Module)
            # pylint: disable=protected-access
            mod._scan_fields(mod.__class__.__dict__.keys())
            # pylint: disable=protected-access
            mod._scan_fields(mod.__dict__.keys())
            return mod

        self.apply(_scan_apply_fn)
        return self
