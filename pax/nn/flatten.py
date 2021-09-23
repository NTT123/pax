"""Flatten a module"""
from typing import List

import jax
import jax.numpy as jnp
import jmp
from jaxlib.xla_extension import PyTreeDef

from .. import ctx
from ..module import Module, PaxFieldKind, T


class FlattenModule(Module):
    """Flatten a module to a make all its parameters and states as lists of `ndarray`'s."""

    params_leaves: List[jnp.ndarray]
    states_leaves: List[jnp.ndarray]
    params_treedef: PyTreeDef
    states_treedef: PyTreeDef
    module_treedef: PyTreeDef

    def __init__(self, mod: Module):
        """Create a flatten version of the input module."""
        super().__init__()

        params_leaves, params_treedef = jax.tree_flatten(
            mod.filter(PaxFieldKind.PARAMETER)
        )
        states_leaves, states_treedef = jax.tree_flatten(mod.filter(PaxFieldKind.STATE))

        self.params_treedef = params_treedef
        self.states_treedef = states_treedef
        self.module_treedef = jax.tree_structure(mod)
        self.register_parameter_subtree("params_leaves", params_leaves)
        self.register_state_subtree("states_leaves", states_leaves)
        self.num_leaves = len(jax.tree_leaves(mod))

        if hasattr(mod, "unflatten"):
            raise RuntimeError("Cannot flatten a module twice!")

        if not hasattr(mod, "__call__"):
            raise ValueError("Expecting a callable module.")

    def unflatten(self):
        """Recreate the original module."""
        params = jax.tree_unflatten(self.params_treedef, self.params_leaves)
        states = jax.tree_unflatten(self.states_treedef, self.states_leaves)
        module = jax.tree_unflatten(self.module_treedef, [0] * self.num_leaves)
        module = module.update(params)
        module = module.update(states)
        return module

    def __call__(self, *args, **kwargs):
        """Recreate the original module, then call it."""
        module = self.unflatten()
        out = module(*args, **kwargs)

        with ctx.mutable():
            states_leaves, _ = jax.tree_flatten(module.filter(PaxFieldKind.STATE))
        self.states_leaves = states_leaves
        return out

    def freeze(self: T) -> T:
        """Disabled in FlattenModule"""
        raise RuntimeError("Disabled in FlattenModule")

    def unfreeze(self: T) -> T:
        """Disabled in FlattenModule"""
        raise RuntimeError("Disabled in FlattenModule")

    def train(self: T, mode: bool = True):
        """Disabled in FlattenModule"""
        raise RuntimeError("Disabled in FlattenModule")

    def eval(self: T) -> T:
        """Disabled in FlattenModule"""
        raise RuntimeError("Disabled in FlattenModule")

    def __repr__(self) -> str:
        s = self.unflatten().__repr__()
        return f"Flatten({s})"
