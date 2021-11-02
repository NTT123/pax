"""Build a module from a directed graph"""

from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp

from .module import Module
from .pure import pure
from .utility_modules import Lambda


def identity(x):
    return x


@dataclass(repr=False)
class Node:
    """A node that stores its parents nodes,
    a function and the output value of the function."""

    parents: Tuple["Node", ...]
    fx: Union[Module, Callable]
    value: jnp.ndarray

    def __post_init__(self):
        if not isinstance(self.fx, Module):
            self.fx = Lambda(self.fx)

    def __rshift__(self, fn):
        """Create a new node.

        Example:
        >>> import jax, pax, jax.numpy as jnp
        >>> from functools import partial
        >>> x = pax.InputNode(jnp.array(1.))
        >>> y = x >> partial(jax.lax.add, 1.)
        """
        return Node((self,), fn, pure(fn)(self.value))

    def __and__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a tuple.

        Example:

        >>> x = pax.InputNode(1)
        >>> y = pax.InputNode(2)
        >>> z = x & y
        """
        if isinstance(other, CatNode):
            return CatNode((self, *other.parents), identity, (self.value, *other.value))
        else:
            return CatNode((self, other), identity, (self.value, other.value))

    def __or__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a multi args node.

        Example:

        >>> x = pax.InputNode(1)
        >>> y = pax.InputNode(2)
        >>> z = x | y
        """
        if isinstance(other, ArgsNode):
            return ArgsNode(
                (self, *other.parents), identity, (self.value, *other.value)
            )
        else:
            return ArgsNode((self, other), identity, (self.value, other.value))

    def binary_ops(self, fn, other):
        """Create a new using a binary operator."""

        @wraps(fn)
        def bin_ops_fn(xs):
            return fn(xs[0], xs[1])

        return Node((self, other), bin_ops_fn, fn(self.value, other.value))

    @property
    def shape(self):
        if hasattr(self, "value") and self.value is not None:
            return self.value.shape
        else:
            return None

    @property
    def dtype(self):
        if hasattr(self, "value") and self.value is not None:
            return self.value.dtype
        else:
            return None

    def __eq__(self, o: object) -> bool:
        return self is o

    def __hash__(self) -> int:
        return id(self)

    def __deepcopy__(self, _):
        raise TypeError("DO NOT COPY")

    def __copy__(self):
        raise TypeError("DO NOT COPY")


class InputNode(Node):
    """An input node. It is NOT a ndarray."""

    parents: Tuple[Node, ...]
    fx: Union[Module, Callable]
    value: jnp.ndarray

    def __init__(self, value: jnp.ndarray, fx=lambda x: x):
        super().__init__((), fx, value)


class CatNode(Node):
    """Concatenate two nodes to create a new "tuple" node."""

    def __rshift__(self, fn):
        return Node(self.parents, fn, fn(self.value))

    def __and__(self, other: Node):
        if isinstance(other, CatNode):
            return CatNode(
                (*self.parents, *other.parents),
                identity,
                (*self.value, *other.value),
            )
        else:
            return CatNode((*self.parents, other), identity, (*self.value, other.value))


class ArgsNode(Node):
    """Concatenate two nodes to create a multi output node."""

    def __rshift__(self, fn):
        """Create a new node with multiple arguments."""

        @wraps(fn)
        def multi_args_fn(xs):
            return fn(*xs)

        assert isinstance(self.value, tuple)
        return Node((self,), multi_args_fn, multi_args_fn(self.value))

    def __or__(self, other: Node):
        if isinstance(other, ArgsNode):
            return ArgsNode(
                (*self.parents, *other.parents),
                identity,
                (*self.value, *other.value),
            )
        else:
            return ArgsNode(
                (*self.parents, other), identity, (*self.value, other.value)
            )


class GraphModule(Module):
    """A module that uses a directed graph to represent its computation."""

    output_node: Node
    modules: List[Module]

    def __init__(self, inputs, output, name: Optional[str] = None):
        super().__init__(name=name)

        self.register_module("modules", [])

        @lru_cache(maxsize=None)
        def transform(node: Node):
            p = tuple(transform(parent) for parent in node.parents)
            if node in inputs:
                idx = inputs.index(node)
                return InputNode(value=None, fx=lambda xs: xs[idx])
            elif isinstance(node.fx, Module):
                mod_idx = len(self.modules)
                self.modules.append(node.fx)
                fx = lambda mods, xs: mods[mod_idx](xs)
                fx.__name__ = f"F[{mod_idx}]"
                return Node(p, fx, None)
            else:
                raise RuntimeError("Impossible")

        self.output_node = transform(output)

    def __call__(self, *xs):
        @lru_cache(maxsize=None)
        def run(node: Node):
            if isinstance(node, InputNode):
                return node.fx(xs)
            else:
                ii = tuple(run(p) for p in node.parents)
                if len(node.parents) == 1:
                    (ii,) = ii
                return node.fx(self.modules, ii)

        return run(self.output_node)


def build_graph_module(func):
    """Build a graph module from a forward function.

    Example:

    >>> def residual_forward(x):
    ...     y = x >> pax.nn.Linear(x.shape[-1], x.shape[-1])
    ...     y >>= jax.nn.relu
    ...     z = (x & y) >> (lambda xs: xs[0] + xs[1])
    ...     return z
    ...
    >>> net = pax.build_graph_module(residual_forward)(jnp.empty((3, 8)))
    """

    def builder_fn(*inputs):
        inputs = tuple(InputNode(x) for x in inputs)
        output = func(*inputs)
        assert isinstance(output, Node)
        mod = GraphModule(inputs, output)
        del inputs, output
        return mod

    return builder_fn
