"""A model as a graph of nodes and Module arrows"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Tuple, Dict

import jax
import pax
from pax import Module
import jax.numpy as jnp


pax.seed_rng_key(42)


def identity(x):
    return x


@dataclass(repr=False)
class NodeBase:
    parents: Tuple["Node", ...]
    fx: Callable  # = field(repr=False)
    value: jnp.ndarray

    def __rshift__(self, fn):
        return Node((self,), fn, fn(self.value))


class Node(NodeBase):
    def __repr__(self):
        return f"{self.__class__.__qualname__}[parents={self.parents}, value_shape={jax.tree_map(lambda x: x.shape, self.value)}, fx={self.fx}]"

    def __and__(self: NodeBase, other: NodeBase):
        if isinstance(other, CatNode):
            return CatNode(
                (self.parents, *other.parents), identity, (self.value, *other.value)
            )
        else:
            return CatNode((self, other), identity, (self.value, other.value))


class InputNode(Node):
    pass


class CatNode(Node):
    def __rshift__(self, fn):
        return Node(self.parents, fn, fn(self.value))

    def __and__(self, other: NodeBase):
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


@dataclass(repr=False)
class GraphModule(pax.Module):
    inputs: tuple
    output: Node
    modules: Dict = field(init=False)

    def __post_init__(self):
        self.modules = {}

        def make(node: Node):
            f = node.fx
            p = tuple(make(parent) for parent in node.parents)

            if isinstance(f, Module):
                idd = id(f)
                self.modules[idd] = f
                f = lambda xs: self.modules[idd](xs)

            cls = node.__class__
            if node in self.inputs:
                idx = self.inputs.index(node)
                f = lambda xs: xs[idx]

            return cls(tuple(p), f, None)

        self.output = make(self.output)

    def __call__(self, *xs):
        def run(node: Node):
            if isinstance(node, InputNode):
                return node.fx(xs)
            else:
                ii = tuple(run(p) for p in node.parents)
                if len(node.parents) == 1:
                    (ii,) = ii
                return node.fx(ii)

        return run(self.output)


def build_graph_module(func):
    def f(*inputs):
        inputs = jax.tree_map(lambda x: InputNode((), lambda v: v, x), inputs)
        output = func(*inputs)
        return GraphModule(inputs, output)

    return f


def forward(x: InputNode):
    y = x >> pax.nn.Linear(3, 3) >> jax.nn.relu
    z = (x & y) >> partial(jnp.concatenate, axis=-1)
    z >>= pax.nn.Linear(6, 3)
    z >>= partial(jax.nn.log_softmax, axis=-1)
    return z


x = jnp.ones((5, 3))
net = build_graph_module(forward)(x)
print(net.summary())
# GraphModule
# ├── Linear[in_dim=6, out_dim=3, with_bias=True]
# └── Linear[in_dim=3, out_dim=3, with_bias=True]
