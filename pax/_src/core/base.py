"""PAX BaseModule."""

# Note: This file is originated from
# https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
# which is under MIT License.

import functools
from copy import deepcopy
from typing import Any, List, Mapping, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

# pylint: disable=no-name-in-module
from jaxlib.xla_extension import CompiledFunction

from .threading_local import is_deep_copy_enabled

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")


class BaseModule:
    """BaseModule manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    BaseModule maintains a ``_pytree_attributes`` tuple that lists all subtree attribute names.
    """

    _pytree_attributes: Tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        self._pytree_attributes = ()

    @property
    def pytree_attributes(self):
        return self._pytree_attributes

    def find_and_register_pytree_attributes(self: T):
        """Find and register ndarrays and submodules."""
        pytree_attributes = []
        for name, value in self.__dict__.items():
            leaves, _ = jax.tree_flatten(
                value, is_leaf=lambda x: isinstance(x, (BaseModule, EmptyNode))
            )
            is_pytree = lambda x: isinstance(
                x, (jnp.ndarray, np.ndarray, BaseModule, EmptyNode)
            )
            if any(map(is_pytree, leaves)):
                pytree_attributes.append(name)
        super().__setattr__("_pytree_attributes", tuple(pytree_attributes))

    def tree_flatten(self) -> Tuple[List[jnp.ndarray], Mapping[str, Any]]:
        """Convert a module to ``(children, treedef)``."""
        aux = dict(self.__dict__)
        children = [aux.pop(name) for name in self._pytree_attributes]

        if is_deep_copy_enabled():
            leaves, treedef = jax.tree_flatten(aux)
            new_leaves = []
            black_list = (jax.custom_jvp, functools.partial, CompiledFunction)
            for leaf in leaves:
                try:
                    if isinstance(leaf, black_list):
                        new_leaves.append(leaf)
                    else:
                        new_leaf = deepcopy(leaf)
                        new_leaves.append(new_leaf)
                except TypeError:
                    new_leaves.append(leaf)
            aux = jax.tree_unflatten(treedef, new_leaves)

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        module_dict = module.__dict__
        module_dict.update(aux)
        module_dict.update(zip(module._pytree_attributes, children))
        return module

    def __init_subclass__(cls):
        """Any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def __eq__(self, o: object) -> bool:
        """Compare two modules."""
        if id(self) == id(o):
            return True

        if type(self) is not type(o):
            return False

        self_leaves, self_treedef = jax.tree_flatten(self)
        o_leaves, o_treedef = jax.tree_flatten(o)

        if len(self_leaves) != len(o_leaves):
            return False

        if self_treedef != o_treedef:
            return False

        leaves_equal = jax.tree_map(lambda a, b: a is b, self_leaves, o_leaves)
        return all(leaves_equal)

    def __hash__(self) -> int:
        leaves, treedef = jax.tree_flatten(self)
        leaves = jax.tree_map(lambda x: (x.shape, x.dtype), leaves)
        return hash((tuple(leaves), treedef))


@jax.tree_util.register_pytree_node_class
class EmptyNode(Tuple):
    """We use this class to mark deleted nodes.

    Note: this is inspired by treex's `Nothing` class.
    """

    def tree_flatten(self):
        """Flatten empty node."""
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten empty node."""
        del aux, children
        return EmptyNode()
