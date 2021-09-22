"""Flatten a module"""
from typing import Any, Dict, List, Optional, Union

import jax

from ..module import Module, PaxFieldKind, T


class FlattenModule(Module):
    def __init__(self, mod: Module):
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

    def unflatten(self):
        params = jax.tree_unflatten(self.params_treedef, self.params_leaves)
        states = jax.tree_unflatten(self.states_treedef, self.states_leaves)
        module = jax.tree_unflatten(
            self.module_treedef,
            [0] * (len(self.params_leaves) + len(self.states_leaves)),
        )
        module = module.update(params)
        module = module.update(states)
        return module

    def __call__(self, *args, **kwargs):
        module = self.unflatten()
        out = module(*args, **kwargs)

        params_leaves, _ = jax.tree_flatten(module.filter(PaxFieldKind.PARAMETER))
        states_leaves, _ = jax.tree_flatten(module.filter(PaxFieldKind.STATE))
        self.params_leaves = params_leaves
        self.states_leaves = states_leaves
        return out

    def __getattr__(self, name):
        return getattr(self.unflatten(), name)

    def freeze(self: T) -> T:
        raise RuntimeError("Disabled in a FlattenModule")

    def unfreeze(self: T) -> T:
        raise RuntimeError("Disabled in a FlattenModule")

    def train(self: T, mode: bool = True):
        raise RuntimeError("Disabled in a FlattenModule")

    def eval(self: T) -> T:
        raise RuntimeError("Disabled in a FlattenModule")

    def __repr__(self, info: Optional[Dict[str, Any]] = None) -> str:
        return super().__repr__(info=info)

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        return super().summary(return_list=return_list)
