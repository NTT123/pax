from collections import namedtuple
from typing import Any, List, OrderedDict, Sequence

import jax


class Node(object):
    def tree_flatten(self):
        return [], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls()
        return obj

    def __init_subclass__(cls) -> None:
        jax.tree_util.register_pytree_node_class(cls)


jax.tree_util.register_pytree_node_class(Node)


class PaxList(Node, list):
    def tree_flatten(self):
        return [list(self)], None

    @classmethod
    def tree_unflatten(self, aux, children):
        return PaxList(children[0])


class PaxDict(Node, OrderedDict):
    def tree_flatten(self):
        return [OrderedDict(self)], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return PaxDict(children[0])


class Leaf(Node):

    value: Any = None

    def __init__(self, value):
        if isinstance(
            value, (str, frozenset, tuple, int, float, complex, bool, namedtuple)
        ):
            self.value = value
        else:
            raise ValueError(f"A value of type {type(value)} cannot be a leaf")

    def tree_flatten(self):
        return [self.value], None

    def update(self, value):
        self.value = value

    @classmethod
    def tree_unflatten(self, aux, children):
        return Leaf(children[0])

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__str__()


def to_tree(value):
    if isinstance(value, (list)):
        new_value = (to_tree(v) for v in value)
        return PaxList(new_value)
    elif isinstance(value, (dict, OrderedDict)):
        new_value = ((k, to_tree(v)) for k, v in value.items())
        return PaxDict(new_value)
    elif isinstance(value, tuple):
        return tuple(to_tree(v) for v in value)
    else:
        return value


class Tree(Node):
    def __init__(self, root):
        super().__init__()
        self.root = to_tree(root)

    def tree_flatten(self):
        return [self.root], None

    @classmethod
    def tree_unflatten(self, aux, chidlren):
        return Tree(chidlren[0])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root.__repr__()})"


class Parameter(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class State(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class Module(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class Attribute(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class ParameterTree(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class StateTree(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class ModuleTree(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)


class AttributeTree(Tree):
    def __init__(self, root):
        super().__init__(root)
        leaves = jax.tree_leaves(self)
