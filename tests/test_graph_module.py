"""Test graph module"""

from functools import partial

import jax
import jax.numpy as jnp
import pax
import pytest
from pax._src.core.graph_module import build_graph_module
from pax.graph import GraphModule


def test_simple_graph():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    assert y.value.shape == (3, 4)


def test_cat_graph():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    z = x & y
    t = z >> partial(jnp.concatenate, axis=-1)
    assert t.value.shape == (3, 7)


def test_cat_merge_left():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    q = y & y
    z = q & x
    assert z.parents == (y, y, x)


def test_cat_merge_right():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    q = y & y
    z = x & q
    assert z.parents == (x, y, y)


def test_merge_2_cat():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    q = y & y
    t = x & x
    k = q & t
    assert k.parents == (y, y, x, x)


def test_3_cat_graph():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    z = x & y & x
    t = z >> partial(jnp.concatenate, axis=-1)
    assert t.value.shape == (3, 10)


def test_3_cat_graph_module():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    z = x & y & y
    t = z >> partial(jnp.concatenate, axis=-1)
    _ = GraphModule((x,), t)


def test_or_graph():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 3) >> jax.nn.relu
    z = (x | y) >> jax.lax.add
    assert z.value.shape == (3, 3)


def test_merge_2_or():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    q = y | y
    t = x | x
    k = t | q
    assert k.parents == (x, x, y, y)


def test_or_merge_left():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 3) >> jax.nn.relu
    z = x | y
    t = z | x
    assert t.parents == (x, y, x)


def test_or_merge_right():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 3) >> jax.nn.relu
    z = x | y
    t = x | z
    assert t.parents == (x, x, y)


def test_cat_graph_merge():
    x = pax.graph.InputNode(jnp.zeros((3, 3)))
    y = x >> pax.nn.Linear(3, 4) >> jax.nn.relu
    q = y | y
    z = x | q
    assert z.parents == (x, y, y)


def test_binops():
    x = pax.graph.InputNode(jnp.ones((3, 3)))
    y = x.binary_ops(jax.lax.add, x)
    assert y.parents == (x, x)
    assert jnp.array_equal(y.fx((x.value, x.value)), jnp.ones((3, 3)) * 2)
    assert jnp.array_equal(y.value, jnp.ones((3, 3)) * 2)


def test_type_shape():
    x = pax.graph.InputNode(jnp.ones((3, 3), dtype=jnp.int32))
    assert x.shape == (3, 3)
    assert x.dtype == jnp.int32


def test_build_residual_net():
    def residual(x):
        y = x >> pax.nn.Linear(3, 3) >> jax.nn.relu
        t = x >> pax.nn.Linear(3, 3) >> jax.nn.tanh
        z = (y | t) >> jax.lax.add
        return z

    x = jnp.empty((1, 3))
    net = build_graph_module(residual)(x)
    y = net(x)
    assert y.shape == (1, 3)


def test_reuse_module_error():
    def reuse(x):
        mod = pax.nn.Linear(3, 3)
        y = x >> mod >> jax.nn.relu
        t = x >> mod
        z = (y | t) >> jax.lax.add
        return z

    x = jnp.empty((1, 3))
    with pytest.raises(ValueError):
        net = build_graph_module(reuse)(x)
