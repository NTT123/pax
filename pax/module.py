"""Pax module.

Note: This file is originated from https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

import collections
import inspect
from typing import Callable, List, Optional, Type, TypeVar, Union

import jax
import jax.tree_util

from . import tree

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", bound="Module")
FilterFn = Callable[[tree.Leaf, T], bool]


class Module:
    _training = True
    _param_filter_fn: Optional[FilterFn] = None

    @property
    def param_filter_fn(self) -> FilterFn:
        return self._param_filter_fn

    @property
    def training(self) -> bool:
        return self._training

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        _tree = {}
        _not_tree = {}

        for name, value in fields.items():
            # `_training` is already in `props`.
            if name in ["_training"]:
                continue
            annotation = annotations.get(name, None)
            annotation = _flatten_tree_type(annotation)

            if annotation is None or not inspect.isclass(annotation):
                _not_tree[name] = value
            elif issubclass(annotation, Module):
                _tree[name] = value
            elif issubclass(annotation, tree.Leaf):
                _tree[name] = jax.tree_map(
                    lambda x: annotation(x, {"module": self}), value
                )
            else:
                _not_tree[name] = value

        return tuple(_tree.values()), dict(
            tree=_tree.keys(),
            not_tree=_not_tree,
            props=dict(
                _training=self._training,
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

    def filter(self: T, *filters: Type, filter_fn: Optional[FilterFn] = None) -> T:
        """Filtering a module based on the type of leaf nodes or a custom function.

        If `filter_fn` is not None, a filter function will be applied.
        We also pass `self` to `filter_fn` function as this is helpful when selecting a subset of `self`.
        """
        flat: List[tree.Leaf]

        if filter_fn is None:

            def filter_fn(x, m: T):
                return isinstance(x, filters)

        flat, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, tree.Leaf)
        )
        flat_out = [
            leaf.value
            if isinstance(leaf, tree.Leaf) and filter_fn(leaf, self)
            else tree.Nothing()
            for leaf in flat
        ]
        module = jax.tree_unflatten(treedef, flat_out)

        return module

    def update(self: T, other: T, *rest: T) -> T:
        modules = (self, other) + rest

        def merge_fn(xs):
            acc, *xs = xs
            for x in xs:
                if not isinstance(x, tree.Nothing):
                    acc = x
            return acc

        flats, treedefs = zip(
            *[
                jax.tree_flatten(m, is_leaf=lambda x: isinstance(x, tree.Nothing))
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

    def filter_parameters(self, filter_fn: FilterFn):
        """Set a filter function for trainable parameters.
        This function is used to fine-tune a subset of the module.

        Arguments:
            filter_fn: FilterFn, a filter function which picks trainble parameters.

        Returns:
            new_self: a new module.
        """
        new_self = self.copy()
        new_self._param_filter_fn = filter_fn
        return new_self

    def parameters(self):
        """Return trainable parameters of the module.

        Apply `self.param_filter_fn` if available."""
        params = self.filter(tree.Parameter)
        if self.param_filter_fn is not None:
            params = params.filter(filter_fn=self.param_filter_fn)
        return params


def _flatten_tree_type(t: Optional[type]) -> Optional[type]:
    """Flatten Optional[T], List[T] and Sequence[T], Dict[K, T] -> T.

    Note: Tuple is not supported.
    """
    if t is None or inspect.isclass(t):
        return t
    elif hasattr(t, "__origin__") and hasattr(t, "__args__"):
        subtypes = [x for x in t.__args__ if x != type(None)]
        # check if Leaf/Module in Optional type, Union[Leaf, int] is not supported.
        if t.__origin__ is Union:
            if len(subtypes) == 1:
                x = _flatten_tree_type(subtypes[0])
                if inspect.isclass(x) and issubclass(x, (tree.Leaf, Module)):
                    return x
        elif t.__origin__ in [list, collections.abc.Sequence] and len(subtypes) == 1:
            return _flatten_tree_type(subtypes[0])
        elif t.__origin__ is dict:
            [k, v] = t.__args__
            del k
            v = _flatten_tree_type(v)
            return v
    return t
