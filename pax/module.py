"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

import copy
from typing import Any, Callable, Dict, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", bound="Module")
FilterFn = Callable[[Any, T], bool]


class Module:
    _properties: Dict[str, Any]

    def __init__(self):
        self._properties = dict()
        self._properties["_training"] = True
        self._properties["_parameters"] = set()
        self._properties["_states"] = set()
        self._properties["_modules"] = set()
        self._properties["_parameter_subtrees"] = set()
        self._properties["_state_subtrees"] = set()
        self._properties["_module_subtrees"] = set()

    @property
    def training(self) -> bool:
        return self._properties["_training"]

    def register_parameter(self, name: str, value: jnp.ndarray):
        setattr(self, name, value)
        self._properties["_parameters"].add(name)

    def register_state(self, name: str, value: jnp.ndarray):
        setattr(self, name, value)
        self._properties["_states"].add(name)

    def register_module(self, name: str, value: Any):
        setattr(self, name, value)
        self._properties["_modules"].add(name)

    def register_parameter_subtree(self, name: str, value: Any):
        setattr(self, name, value)
        self._properties["_parameter_subtrees"].add(name)

    def register_state_subtree(self, name: str, value: Any):
        setattr(self, name, value)
        self._properties["_state_subtrees"].add(name)

    def register_module_subtree(self, name: str, value: Any):
        setattr(self, name, value)
        self._properties["_module_subtrees"].add(name)

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        _tree = {}
        _not_tree = {}

        all_tree_fields = set.union(
            self._properties["_parameters"],
            self._properties["_states"],
            self._properties["_modules"],
            self._properties["_parameter_subtrees"],
            self._properties["_state_subtrees"],
            self._properties["_module_subtrees"],
        )

        for name, value in fields.items():
            if name == "_properties":
                continue

            if name in all_tree_fields:
                _tree[name] = value
            elif isinstance(value, Module):
                # when a field is Module's instance, it is automatically part of the pytree.
                _tree[name] = value
            else:
                _not_tree[name] = value

        return tuple(_tree.values()), dict(
            tree=_tree.keys(),
            not_tree=_not_tree,
            _properties=copy.deepcopy(self._properties),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)
        module._properties = copy.deepcopy(aux_data["_properties"])

        for i, k in enumerate(aux_data["tree"]):
            setattr(module, k, children[i])

        for k, v in aux_data["not_tree"].items():
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

        param_field_names = self._properties["_parameters"].union(
            self._properties["_parameter_subtrees"]
        )
        state_field_names = self._properties["_states"].union(
            self._properties["_state_subtrees"]
        )
        module_field_names = self._properties["_modules"].union(
            self._properties["_module_subtrees"]
        )

        for name, value in fields.items():
            if name in module_field_names:
                value = jax.tree_map(
                    lambda x: x.filter(keep),
                    value,
                    is_leaf=lambda x: isinstance(x, Module),
                )
            elif name in param_field_names:
                fn1 = lambda x: x
                fn2 = lambda x: None
                fn = fn1 if keep == "parameter" else fn2
                value = jax.tree_map(fn, value)
            elif name in state_field_names:
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
        model._properties["_training"] = mode
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
        model._properties["_states"].update(self._properties["_parameters"])
        model._properties["_parameters"].clear()
        model._properties["_state_subtrees"].update(
            self._properties["_parameter_subtrees"]
        )
        model._properties["_parameter_subtrees"].clear()

        return model

    def hk_init(self, *args, enable_jit=False, **kwargs):
        """Return a new initialized module.

        Arguments:
        enable_jit: bool, if using `jax.jit` for the init function.
        """

        def init_fn(mod, args, kwargs):
            mod = mod.copy()
            mod(*args, **kwargs)
            return mod

        if enable_jit:
            init_fn = jax.jit(init_fn)
        return init_fn(self, args, kwargs)
