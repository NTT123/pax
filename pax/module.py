"""Pax module.

Note: This file is originated from https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
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


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()

    def __repr__(self) -> str:
        return "Nothing"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Nothing)


class Module:
    _training = True
    _module_filter_fn: Optional[FilterFn] = None
    _parameters: Set[str] = set()
    _states: Set[str] = set()
    _param_subtrees: Set[str] = set()
    _state_subtrees: Set[str] = set()
    _module_subtrees: Set[str] = set()
    _modules: Set[str] = set()

    @property
    def module_filter_fn(self) -> FilterFn:
        return self._module_filter_fn

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
                "_module_filter_fn",
                "_parameters",
                "_states",
                "_param_subtrees",
                "_state_subtrees",
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
                _module_filter_fn=self._module_filter_fn,
                _parameters=self._parameters,
                _states=self._states,
                _param_subtrees=self._param_subtrees,
                _state_subtrees=self._state_subtrees,
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
            setattr(module, k, v)

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def filter(
        self: T,
        filter_type: str = "parameter",
        module_filter_fn: Optional[FilterFn] = None,
    ) -> T:
        """Filtering a module based on the type of leaf nodes or a custom function.

        If `filter_fn` is not None, a filter function will be applied.
        We also pass `self` to `filter_fn` function as this is helpful when selecting a subset of `self`.
        """
        flat: List[Any]
        fields = vars(self)
        cls = self.__class__
        module = cls.__new__(cls)

        if module_filter_fn is None:
            module_filter_fn = lambda x, info: x

        for name, value in fields.items():
            if name in self._module_subtrees or name in self._modules:
                value = jax.tree_map(
                    lambda x: module_filter_fn(
                        x.filter(filter_type, module_filter_fn),
                        {"old": x, "parent": self, "name": name},
                    ),
                    value,
                    is_leaf=lambda x: isinstance(x, Module),
                )
            elif name in self._param_subtrees or name in self._parameters:
                if module_filter_fn is None:
                    module_filter_fn = lambda x, y: True
                fn1 = lambda x: x
                fn2 = lambda x: Nothing()
                fn = fn1 if filter_type == "parameter" else fn2
                value = jax.tree_map(fn, value)
            elif name in self._state_subtrees or name in self._states:
                fn1 = lambda x: x
                fn2 = lambda x: Nothing()
                fn = fn1 if filter_type == "state" else fn2
                value = jax.tree_map(fn, value)
            elif isinstance(value, Module):
                value = module_filter_fn(
                    value.filter(filter_type, module_filter_fn),
                    {"old": value, "parent": self, "name": name},
                )
            setattr(module, name, value)

        return module

    def update(self: T, other: T, *rest: T) -> T:
        modules = (self, other) + rest

        def merge_fn(xs):
            acc, *xs = xs
            for x in xs:
                if not isinstance(x, Nothing):
                    acc = x
            return acc

        flats, treedefs = zip(
            *[
                jax.tree_flatten(m, is_leaf=lambda x: isinstance(x, Nothing))
                for m in modules
            ]
        )
        # flat_out = jax.tree_util.tree_map(merge_fn, *flats)
        flat_out = [merge_fn(values) for values in zip(*flats)]
        module = jax.tree_unflatten(treedefs[0], flat_out)

        return module

    def train(self: T, mode=True):
        """Rebuild a new model recursively and set `self._training = mode`."""
        submods, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        new_submodes = []
        for mod in submods:
            if isinstance(mod, Module):
                new_submodes.append(mod.train(mode=mode))
            else:
                new_submodes.append(mod)
        model = jax.tree_unflatten(treedef, new_submodes)
        model._training = mode
        return model

    def eval(self: T) -> T:
        return self.train(False)

    def filter_modules(self, filter_fn: FilterFn):
        """Set a filter function for trainable modules.
        This function is used to fine-tune a subset of the module.

        Arguments:
            filter_fn: FilterFn, a filter function which picks trainble parameters.

        Returns:
            new_self: a new module.
        """
        new_self = self.copy()
        new_self._module_filter_fn = filter_fn
        return new_self

    def parameters(self):
        """Return trainable parameters of the module.

        Apply `self.param_filter_fn` if available."""
        params = self.filter("parameter")
        if self.module_filter_fn is not None:
            params = params.filter(module_filter_fn=self.module_filter_fn)
        return params
