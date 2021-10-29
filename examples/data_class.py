"""How to implement a PAX module using python dataclass."""
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import pax

pax.seed_rng_key(42)

# data
x = jax.random.normal(pax.next_rng_key(), (32, 1))
y = jax.random.normal(pax.next_rng_key(), (32, 1))


@dataclass
class Linear(pax.ParameterModule):
    """A Linear module"""

    in_dim: int
    out_dim: int
    with_bias: bool = True
    name: Optional[str] = None
    weight: jnp.ndarray = field(init=False, repr=False)
    bias: Optional[jnp.ndarray] = field(init=False, repr=False)

    def __post_init__(self):
        self.weight = jax.random.normal(pax.next_rng_key(), (self.in_dim, self.out_dim))
        if self.with_bias:
            self.bias = jax.random.normal(pax.next_rng_key(), (self.out_dim,))
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.dot(x, self.weight)
        if self.with_bias:
            x = x + self.bias
        return x


fc = Linear(3, 4, name="fc1")
print(fc.summary())

x = jnp.ones((32, 3))
y = fc(x)
assert y.shape == (32, 4)
