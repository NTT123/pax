"""PAX BaseModule."""

# Note: This file is originated from
# https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
# which is under MIT License.

from typing import Any, List, Mapping, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")


class BaseModule:
    """BaseModule manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    BaseModule maintains a ``pytree_attributes`` tuple that lists all subtree attribute names.
    """

    _pytree_attributes: Tuple[str, ...] = ()
    _mixed_pytree_attributes: Optional[Tuple[str, ...]] = None

    @property
    def pytree_attributes(self):
        if self._mixed_pytree_attributes is not None:
            return self._pytree_attributes + self._mixed_pytree_attributes
        else:
            return self._pytree_attributes

    def find_and_register_pytree_attributes(self: T):
        """Find and register ndarrays and submodules."""
        is_mod_or_node = lambda x: isinstance(x, (BaseModule, EmptyNode))
        is_pytree = lambda x: isinstance(x, pytree_cls)

        pytree_attributes = []
        mixed_pytree_attributes = []
        for name, value in self.__dict__.items():
            leaves, _ = jax.tree_flatten(value, is_leaf=is_mod_or_node)
            pytree_cls = (jnp.ndarray, np.ndarray, BaseModule, EmptyNode)
            any_pytree = any(map(is_pytree, leaves))
            all_pytree = all(map(is_pytree, leaves))
            if any_pytree and all_pytree:
                pytree_attributes.append(name)
            elif any_pytree:
                mixed_pytree_attributes.append(name)
        self._pytree_attributes = tuple(pytree_attributes)
        if len(mixed_pytree_attributes) > 0:
            self._mixed_pytree_attributes = tuple(mixed_pytree_attributes)
        else:
            self._mixed_pytree_attributes = None

    def tree_flatten(self) -> Tuple[List[jnp.ndarray], Mapping[str, Any]]:
        """Convert a module to ``(children, treedef)``."""
        aux = dict(self.__dict__)
        children = [aux.pop(name) for name in self._pytree_attributes]
        if self._mixed_pytree_attributes is not None:
            is_module = lambda x: isinstance(x, BaseModule)
            array_mod_cls = (jnp.ndarray, np.ndarray, BaseModule)
            is_array_mod = lambda x: isinstance(x, array_mod_cls)
            for name in self._mixed_pytree_attributes:
                value = aux.pop(name)
                leaves, treedef = jax.tree_flatten(value, is_leaf=is_module)
                leaves = (v if is_array_mod(v) else ValueNode(v) for v in leaves)
                value = jax.tree_unflatten(treedef, leaves)
                children.append(value)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        module_dict = module.__dict__
        module_dict.update(aux)
        module_dict.update(zip(module._pytree_attributes, children))
        if module._mixed_pytree_attributes is not None:
            L = len(module._pytree_attributes)
            is_leaf = lambda x: isinstance(x, (ValueNode, BaseModule))
            unwrap = lambda x: x.value if isinstance(x, ValueNode) else x
            for name, value in zip(module._mixed_pytree_attributes, children[L:]):
                module_dict[name] = jax.tree_map(unwrap, value, is_leaf=is_leaf)
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


#  Note: this class is inspired by treex's `Nothing` class.
@jax.tree_util.register_pytree_node_class
class EmptyNode:
    """Mark an uninitialized or deleted pytree node."""

    def tree_flatten(self):
        """Flatten empty node."""
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten empty node."""
        del aux, children
        return EmptyNode()

    def __repr__(self) -> str:
        return "EmptyNode"

    def __eq__(self, o: object) -> bool:
        if isinstance(o, EmptyNode):
            return True
        return False


@jax.tree_util.register_pytree_node_class
class ValueNode:
    """We use this class to store a value in treedef."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def tree_flatten(self):
        return (), self.value

    @classmethod
    def tree_unflatten(cls, value, children):
        return ValueNode(value)

    def __repr__(self) -> str:
        return f"ValueNode({self.value})"
