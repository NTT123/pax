"""A model as a directed graph."""

import jax
import jax.numpy as jnp
import pax
from pax.experimental.graph import Node, build_graph_module

pax.seed_rng_key(42)


def residual_net(x: Node):
    _, D = x.shape
    y = x >> pax.Linear(D, D) >> jax.nn.relu >> pax.Linear(D, D) >> pax.Dropout(0.2)
    z = (x | y) >> jax.lax.add
    return z


inputs = jnp.ones((3, 8))
net = build_graph_module(residual_net)(inputs)
print(net.summary())
net, _ = pax.purecall(net, inputs)
