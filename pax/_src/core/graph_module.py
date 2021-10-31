"""Build a module from a directed graph"""

from dataclasses import dataclass
from functools import partialmethod, wraps
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .module import Module, PaxKind
from .pure import pure
from .utility_modules import Lambda


def identity(x):
    return x


def get_input(xs, index):
    return xs[index]


@dataclass(repr=False)
class Node(Module):
    """A node that stores its parents nodes,
    a function and the output value of the function."""

    parents: Tuple["Node", ...]
    fx: Union[Module, Callable]
    value: jnp.ndarray

    def __post_init__(self):
        self.set_attribute_kind(parents=PaxKind.MODULE, value=PaxKind.STATE)

        if not isinstance(self.fx, Module):

            self.fx = Lambda(self.fx)

        self.set_attribute_kind(fx=PaxKind.MODULE)

    def __rshift__(self, fn):
        """Create a new node.

        Example:
        >>> import jax, pax, jax.numpy as jnp
        >>> from functools import partial
        >>> x = pax.InputNode(value=jnp.array(1.))
        >>> y = x >> partial(jax.lax.add, 1.)
        """
        return Node((self,), fn, fn(self.value))

    def __and__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a tuple.

        Example:

        >>> x = pax.InputNode(1)
        >>> y = pax.InputNode(2)
        >>> z = x & y
        """
        if isinstance(other, CatNode):
            return CatNode(
                (self.parents, *other.parents), identity, (self.value, *other.value)
            )
        else:
            return CatNode((self, other), identity, (self.value, other.value))

    def _binary_ops(self, other, fn):
        """Create a new using a binary operator."""

        @wraps(fn)
        def bin_ops_fn(xs):
            return fn(xs[0], xs[1])

        return Node((self, other), bin_ops_fn, fn(self.value, other.value))

    __add__ = partialmethod(_binary_ops, fn=jax.lax.add)
    __sub__ = partialmethod(_binary_ops, fn=jax.lax.sub)
    __mul__ = partialmethod(_binary_ops, fn=jax.lax.mul)
    __div__ = partialmethod(_binary_ops, fn=jax.lax.div)


@dataclass(repr=False)
class InputNode(Node):
    """An input node. It is NOT a ndarray."""

    parents: Tuple[Node, ...] = ()
    fx: Union[Module, Callable] = lambda x: x
    value: jnp.ndarray = None

    @property
    def shape(self):
        if self.value is not None:
            return self.value.shape
        else:
            return None

    @property
    def dtype(self):
        if self.value is not None:
            return self.value.dtype
        else:
            return None


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
            return CatNode(
                (*self.parents, other.parents), identity, (*self.value, other.value)
            )


class GraphModule(Module):
    """A module that uses a directed graph to represent its computation."""

    output_node: Node

    def __init__(self, inputs, output, name: Optional[str] = None):
        super().__init__(name=name)

        def make(node: Node):
            p = tuple(make(parent) for parent in node.parents)
            if node in inputs:
                idx = inputs.index(node)
                return InputNode((), Lambda(lambda xs: xs[idx], f"get<{idx}>"), None)
            return Node(p, node.fx, None)

        self.output_node = make(output)

    def __call__(self, *xs):
        def run(node: Node):
            if isinstance(node, InputNode):
                return node.fx(xs)
            else:
                ii = tuple(run(p) for p in node.parents)
                if len(node.parents) == 1:
                    (ii,) = ii
                return node.fx(ii)

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

    @pure
    def _func(*inputs):
        inputs = tuple(InputNode((), lambda q: q, x) for x in inputs)
        output = func(*inputs)
        mod = GraphModule(inputs, output)
        del inputs, output
        return mod

    return _func
