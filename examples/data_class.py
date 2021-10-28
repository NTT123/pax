import pax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class Linear(pax.ParameterModule):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __call__(self, x: jnp.ndarray):
        return x * self.weight + self.bias


f = Linear(weight=jnp.array(2.0), bias=jnp.array(1.0))
print(f.summary())
x = jnp.array(0.5)
y = f(x)
print(x, y)
