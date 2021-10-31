"""A model as a directed graph."""

import jax
import pax
import jax.numpy as jnp

pax.seed_rng_key(42)


def residual_net(x: pax.InputNode):
    y = x >> pax.nn.Linear(x.shape[-1], x.shape[-1])
    y >>= jax.nn.relu
    z = x + y
    z >>= pax.nn.Dropout(0.2)
    return z


x = jnp.ones((3, 8))
net = pax.build_graph_module(residual_net)(x)
print(net.summary())
net, y = pax.module_and_value(net)(x)
