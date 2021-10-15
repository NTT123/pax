"""Transform a module to a new one."""
from typing import Any, Generic, List, TypeVar

import jax
import jax.numpy as jnp
from jaxlib.xla_extension import PyTreeDef  # pylint: disable=no-name-in-module

from .module import Module
from .transforms import select_parameters, select_states, update_pytree

TreeDef = Any

T = TypeVar("T", bound=Module)
K = TypeVar("K", bound=Module)
O = TypeVar("O", bound=Module)


class flatten_module(Module, Generic[T]):  # pylint: disable=invalid-name
    """Flatten a module.

    Flatten all parameters and states to lists of `ndarray`'s."""

    params_leaves: List[jnp.ndarray]
    states_leaves: List[jnp.ndarray]
    params_treedef: PyTreeDef
    states_treedef: PyTreeDef
    module_treedef: PyTreeDef

    def __init__(self, mod: T):
        """Create a flatten version of the input module."""
        super().__init__()

        params_leaves, params_treedef = jax.tree_flatten(select_parameters(mod))
        states_leaves, states_treedef = jax.tree_flatten(select_states(mod))

        self.params_treedef = params_treedef
        self.states_treedef = states_treedef
        self.module_treedef = jax.tree_structure(mod)
        self.register_parameters("params_leaves", params_leaves)
        self.register_states("states_leaves", states_leaves)
        self.num_leaves = len(jax.tree_leaves(mod))

        if hasattr(mod, "unflatten"):
            raise RuntimeError("Cannot flatten a module twice!")

        if not hasattr(mod, "__call__"):
            raise ValueError("Expecting a callable module.")

    def unflatten(self) -> T:
        """Recreate the original module."""
        params = jax.tree_unflatten(self.params_treedef, self.params_leaves)
        states = jax.tree_unflatten(self.states_treedef, self.states_leaves)
        module = jax.tree_unflatten(self.module_treedef, [0] * self.num_leaves)
        module = update_pytree(module, other=params)
        module = update_pytree(module, other=states)
        return module

    def __call__(self, *args, **kwargs):
        """Recreate the original module, then call it."""
        module = self.unflatten()
        assert callable(module), "Expecting a callable module." ""

        out = module(*args, **kwargs)

        states_leaves, _ = jax.tree_flatten(select_states(module))
        self.states_leaves = states_leaves
        return out

    def __repr__(self) -> str:
        original_module_repr = self.unflatten().__repr__()
        return f"Flatten({original_module_repr})"

    def eval(self):
        raise RuntimeError("Not supported for a flatten module.")

    def train(self):
        raise RuntimeError("Not supported for a flatten module.")
