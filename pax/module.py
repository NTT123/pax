"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from typing import Any, Callable, NamedTuple, TypeVar, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", bound="Module")
FilterFn = Callable[[Any, T], bool]


class ModuleProperties(NamedTuple):
    training: bool
    parameters: frozenset
    states: frozenset
    modules: frozenset
    parameter_subtrees: frozenset
    state_subtrees: frozenset
    module_subtrees: frozenset


# TODO: use NamedTuple, but, it is slower :-(
ModuleAuxiliaryData = Tuple


class Module:
    _properties: ModuleProperties

    def __init__(self):
        super().__init__()
        self.__dict__["_properties"] = ModuleProperties(
            training=True,
            parameters=frozenset(),
            states=frozenset(),
            modules=frozenset(),
            parameter_subtrees=frozenset(),
            state_subtrees=frozenset(),
            module_subtrees=frozenset(),
        )

    @property
    def training(self) -> bool:
        return self._properties.training

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_properties":
            raise RuntimeError(
                "You SHOULD NOT modify `_properties`. "
                "If you _really_ want to, use `self.__dict__[name] = value` instead."
            )
        self.__dict__[name] = value
        if isinstance(value, Module):
            self.register_module(name, value)

    def register_parameter(self, name: str, value: jnp.ndarray):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            parameters=self._properties.parameters.union([name])
        )

    def register_state(self, name: str, value: jnp.ndarray):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            states=self._properties.states.union([name])
        )

    def register_module(self, name: str, value: Any):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            modules=self._properties.modules.union([name])
        )

    def register_parameter_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            parameter_subtrees=self._properties.parameter_subtrees.union([name])
        )

    def register_state_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            state_subtrees=self._properties.state_subtrees.union([name])
        )

    def register_module_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self.__dict__["_properties"] = self._properties._replace(
            module_subtrees=self._properties.module_subtrees.union([name])
        )

    def tree_flatten(self):
        fields = vars(self)

        _tree = {}
        _not_tree = {}

        all_tree_fields = frozenset.union(
            self._properties.parameters,
            self._properties.states,
            self._properties.modules,
            self._properties.parameter_subtrees,
            self._properties.state_subtrees,
            self._properties.module_subtrees,
        )

        for name, value in fields.items():
            if name == "_properties":
                continue
            elif name in all_tree_fields:
                _tree[name] = value
            else:
                _not_tree[name] = value

        return _tree.values(), (_tree.keys(), _not_tree, self._properties)

    @classmethod
    def tree_unflatten(cls, aux_data: ModuleAuxiliaryData, children):
        module = cls.__new__(cls)
        _tree, _not_tree, _properties = aux_data

        module.__dict__["_properties"] = _properties

        for k, v in _not_tree.items():
            module.__dict__[k] = v

        for i, k in enumerate(_tree):
            module.__dict__[k] = children[i]

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

        param_field_names = (
            self._properties.parameters | self._properties.parameter_subtrees
        )
        state_field_names = self._properties.states | self._properties.state_subtrees
        module_field_names = self._properties.modules | self._properties.module_subtrees

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
            module.__dict__[name] = value

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
        model.__dict__["_properties"] = model._properties._replace(training=mode)
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
        model.__dict__["_properties"] = model._properties._replace(
            states=(model._properties.states | self._properties.parameters)
        )
        model.__dict__["_properties"] = model._properties._replace(
            parameters=frozenset()
        )
        model.__dict__["_properties"] = model._properties._replace(
            state_subtrees=(
                model._properties.state_subtrees | self._properties.parameter_subtrees
            )
        )
        model.__dict__["_properties"] = model._properties._replace(
            parameter_subtrees=frozenset()
        )

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
