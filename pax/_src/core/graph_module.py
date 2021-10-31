"""Build a module from a directed graph"""

from dataclasses import dataclass
from functools import wraps
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

        if not isinstance(self.fx, Module) and self.fx is not identity:
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

    def __or__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a multi args node.

        Example:

        >>> x = pax.InputNode(1)
        >>> y = pax.InputNode(2)
        >>> z = x | y
        """
        if isinstance(other, ArgsNode):
            return ArgsNode(
                (self.parents, *other.parents), identity, (self.value, *other.value)
            )
        else:
            return ArgsNode((self, other), identity, (self.value, other.value))

    def binary_ops(self, fn, other):
        """Create a new using a binary operator."""

        @wraps(fn)
        def bin_ops_fn(xs):
            return fn(xs[0], xs[1])

        return Node((self, other), bin_ops_fn, fn(self.value, other.value))

    # __add__ = partialmethod(binary_ops, jax.lax.add)
    # __sub__ = partialmethod(binary_ops, jax.lax.sub)
    # __mul__ = partialmethod(binary_ops, jax.lax.mul)
    # __div__ = partialmethod(binary_ops, jax.lax.div)

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


@dataclass(repr=False)
class InputNode(Node):
    """An input node. It is NOT a ndarray."""

    parents: Tuple[Node, ...] = ()
    fx: Union[Module, Callable] = lambda x: x
    value: jnp.ndarray = None


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
                (*self.parents, other.parents), identity, (*self.value, other.value)
            )


class GraphModule(Module):
    """A module that uses a directed graph to represent its computation."""

    output_node: Node

    def __init__(self, inputs, output, name: Optional[str] = None):
        super().__init__(name=name)

        def _check_shared_parameters(mod: Node):
            def fx_filter(n: Node):
                if isinstance(n, Node):
                    return Node(n.parents, n.fx, None)
                else:
                    return n

            mod = mod.apply(fx_filter)
            leaves = jax.tree_leaves(jax.tree_map(id, mod))
            if len(leaves) != len(set(leaves)):
                raise ValueError(
                    "Shared parameters (or modules) are not allowed in a GraphModule. "
                    "Please use normal module instead."
                )

        _check_shared_parameters(output)

        def transform_input_nodes(node: Node):
            p = tuple(transform_input_nodes(parent) for parent in node.parents)
            if node in inputs:
                idx = inputs.index(node)
                return InputNode((), Lambda(lambda xs: xs[idx], f"get<{idx}>"), None)
            return Node(p, node.fx, None)

        self.output_node = transform_input_nodes(output)

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
