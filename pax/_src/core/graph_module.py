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
    """A node is an object that stores:

    - parents nodes,
    - a PAX module (or a function),
    - and a value.

    For example:

    >>> x = pax.experimental.graph.Node((), lambda x: x, jnp.array(0))
    >>> x.parents, x.fx, x.value
    ((), Lambda..., DeviceArray(0, dtype=int32, weak_type=True))
    """

    parents: Tuple["Node", ...]
    fx: Union[Module, Callable]
    value: jnp.ndarray

    def __post_init__(self):
        if not isinstance(self.fx, Module):
            self.fx = Lambda(self.fx)

    def __rshift__(self, fn):
        """Create a new node by applying `fn` to the node's value.

        Example:

        >>> import jax, pax, jax.numpy as jnp
        >>> from functools import partial
        >>> x = pax.experimental.graph.Node((), lambda x: x, jnp.array(1.))
        >>> y = x >> partial(jax.lax.add, 1.)
        >>> y.value
        DeviceArray(2., dtype=float32, weak_type=True)
        """
        return Node((self,), fn, pure(fn)(self.value))

    def __and__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a tuple.

        Example:

        >>> x = pax.experimental.graph.InputNode(1)
        >>> y = pax.experimental.graph.InputNode(2)
        >>> z = x & y
        >>> z.value
        (1, 2)
        """
        if isinstance(other, CatNode):
            return CatNode((self, *other.parents), identity, (self.value, *other.value))
        else:
            return CatNode((self, other), identity, (self.value, other.value))

    def __or__(self: "Node", other: "Node"):
        """Concatenate two nodes to create a multi args node.

        Example:

        >>> x = pax.experimental.graph.InputNode(1)
        >>> y = pax.experimental.graph.InputNode(2)
        >>> z = (x | y) >> jax.lax.add
        >>> z.value
        DeviceArray(3, dtype=int32, weak_type=True)
        """
        if isinstance(other, ArgsNode):
            return ArgsNode(
                (self, *other.parents), identity, (self.value, *other.value)
            )
        else:
            return ArgsNode((self, other), identity, (self.value, other.value))

    def binary_ops(self, fn, other):
        """Create a new node using a binary operator.

        Example:

        >>> x = pax.experimental.graph.InputNode(1)
        >>> y = pax.experimental.graph.InputNode(2)
        >>> z = x.binary_ops(jax.lax.sub, y)
        >>> z.value
        DeviceArray(-1, dtype=int32, weak_type=True)
        """

        @wraps(fn)
        def bin_ops_fn(xs):
            return fn(xs[0], xs[1])

        return Node((self, other), bin_ops_fn, fn(self.value, other.value))

    @property
    def shape(self):
        """Return the shape of value.

        Example:

        >>> x = pax.experimental.graph.InputNode(jnp.empty((3, 4)))
        >>> x.shape
        (3, 4)
        """
        if hasattr(self, "value") and self.value is not None:
            return self.value.shape
        else:
            return None

    @property
    def dtype(self):
        """Return dtype of value.

        Example:

        >>> x = pax.experimental.graph.InputNode(jnp.empty((1,), dtype=jnp.int32))
        >>> x.dtype
        dtype('int32')
        """
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
    """An InputNode object represents an input argument of a GraphModule."""

    parents: Tuple[Node, ...]
    fx: Union[Module, Callable]
    value: jnp.ndarray

    def __init__(self, value: jnp.ndarray, fx=lambda x: x):
        """Creata an InputNode object from a value."""
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

        self.modules = []

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
    ...     y = x >> pax.Linear(x.shape[-1], x.shape[-1])
    ...     y >>= jax.nn.relu
    ...     z = (x | y) >> jax.lax.add
    ...     return z
    ...
    >>> from pax.experimental.graph import build_graph_module
    >>> net = build_graph_module(residual_forward)(jnp.empty((3, 8)))
    >>> print(net.summary())
    GraphModule
    ├── Linear(in_dim=8, out_dim=8, with_bias=True)
    ├── x => relu(x)
    ├── x => identity(x)
    └── x => add(x)
    """

    def builder_fn(*inputs):
        inputs = tuple(InputNode(x) for x in inputs)
        output = func(*inputs)
        assert isinstance(output, Node)
        mod = GraphModule(inputs, output)
        del inputs, output
        return mod

    return builder_fn
