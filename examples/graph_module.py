"""A model as a directed graph."""

import jax
import pax
import jax.numpy as jnp
from pax.graph import Node, build_graph_module

pax.seed_rng_key(42)


def residual_net(x: Node):
    y = x >> pax.nn.Linear(x.shape[-1], x.shape[-1])
    y >>= jax.nn.relu
    z = (x | y) >> jax.lax.add
    z >>= pax.nn.Dropout(0.2)
    return z


inputs = jnp.ones((3, 8))
net = build_graph_module(residual_net)(inputs)
print(net.summary())
net, y = pax.module_and_value(net)(inputs)
