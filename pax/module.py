"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from typing import Any, Callable, List, Optional, Set, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", bound="Module")
FilterFn = Callable[[Any, T], bool]


class Module:
    _training = True
    _parameters: Set[str] = set()
    _states: Set[str] = set()
    _param_subtrees: Set[str] = set()
    _state_subtrees: Set[str] = set()
    _module_subtrees: Set[str] = set()
    _modules: Set[str] = set()

    @property
    def training(self) -> bool:
        return self._training

    def register_parameter(self, name: str, value: jnp.ndarray):
        if len(self._parameters) == 0:
            self._parameters = set()
        setattr(self, name, value)
        self._parameters.add(name)

    def register_state(self, name: str, value: jnp.ndarray):
        if len(self._states) == 0:
            self._states = set()
        setattr(self, name, value)
        self._states.add(name)

    def register_param_subtree(self, name: str, value: Any):
        if len(self._param_subtrees) == 0:
            self._param_subtrees = set()
        setattr(self, name, value)
        self._param_subtrees.add(name)

    def register_module_subtree(self, name: str, value: Any):
        if len(self._module_subtrees) == 0:
            self._module_subtrees = set()
        setattr(self, name, value)
        self._module_subtrees.add(name)

    def register_module_subtree(self, name: str, value: Any):
        if len(self._module_subtrees) == 0:
            self._module_subtrees = set()
        setattr(self, name, value)
        self._module_subtrees.add(name)

    def register_state_subtree(self, name: str, value: Any):
        if len(self._state_subtrees) == 0:
            self._state_subtrees = set()
        setattr(self, name, value)
        self._state_subtrees.add(name)

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        _tree = {}
        _not_tree = {}

        for name, value in fields.items():
            # `_training` is already in `props`.
            if name in [
                "_training",
                "_parameters",
                "_states",
                "_modules",
                "_param_subtrees",
                "_state_subtrees",
                "_module_subtrees",
            ]:
                continue

            if name in self._module_subtrees or name in self._modules:
                _tree[name] = value
            elif name in self._param_subtrees or name in self._parameters:
                _tree[name] = value
            elif name in self._state_subtrees or name in self._states:
                _tree[name] = value
            elif isinstance(value, Module):
                # when a field is module, it is automatically part of the pytree.
                _tree[name] = value
            else:
                _not_tree[name] = value

        return tuple(_tree.values()), dict(
            tree=_tree.keys(),
            not_tree=_not_tree,
            props=dict(
                _training=self._training,
                _parameters=set(self._parameters),
                _states=set(self._states),
                _modules=set(self._modules),
                _param_subtrees=set(self._param_subtrees),
                _state_subtrees=set(self._state_subtrees),
                _module_subtrees=set(self._module_subtrees),
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for i, k in enumerate(aux_data["tree"]):
            setattr(module, k, children[i])

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for k, v in aux_data["props"].items():
            if isinstance(v, set):
                setattr(module, k, set(v))
            else:
                setattr(module, k, v)

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def filter(self: T, keep: str = "parameter") -> T:
        """Filtering a module"""
        assert keep in ["parameter", "state"]
        fields = vars(self)
        cls = self.__class__
        module = cls.__new__(cls)

        for name, value in fields.items():
            if name in self._module_subtrees or name in self._modules:
                value = jax.tree_map(
                    lambda x: x.filter(keep),
                    value,
                    is_leaf=lambda x: isinstance(x, Module),
                )
            elif name in self._param_subtrees or name in self._parameters:
                fn1 = lambda x: x
                fn2 = lambda x: None
                fn = fn1 if keep == "parameter" else fn2
                value = jax.tree_map(fn, value)
            elif name in self._state_subtrees or name in self._states:
                fn1 = lambda x: x
                fn2 = lambda x: None
                fn = fn1 if keep == "state" else fn2
                value = jax.tree_map(fn, value)
            elif isinstance(value, Module):
                value = value.filter(keep)
            setattr(module, name, value)

        return module

    def update(self: T, other: T) -> T:
        return jax.tree_map(lambda s, o: (s if o is None else o), self, other)

    def train(self: T, mode=True):
        """Rebuild a new model recursively and set `self._training = mode`."""
        submods, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        new_submods = []
        for mod in submods:
            if isinstance(mod, Module):
                new_submods.append(mod.train(mode=mode))
            else:
                new_submods.append(mod)
        model = jax.tree_unflatten(treedef, new_submods)
        model._training = mode
        return model

    def eval(self: T) -> T:
        return self.train(False)

    def parameters(self):
        """Return trainable parameters of the module.

        Apply `self.param_filter_fn` if available."""
        params = self.filter("parameter")
        return params

    def freeze(self: T) -> T:
        """Convert all trainable parameters to non-trainable states."""
        submods, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        new_submods = []
        for mod in submods:
            if isinstance(mod, Module):
                new_submods.append(mod.freeze())
            else:
                new_submods.append(mod)
        model = jax.tree_unflatten(treedef, new_submods)
        model._states.update(self._parameters)
        model._parameters.clear()
        model._state_subtrees.update(self._param_subtrees)
        model._param_subtrees.clear()

        return model
