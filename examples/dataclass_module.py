"""How to implement a PAX module using python dataclass."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import pax


@dataclass
class Linear(pax.Module):
    """A linear module"""

    in_dim: int
    out_dim: int
    with_bias: bool = True
    name: Optional[str] = None
    weight: jnp.ndarray = field(init=False, repr=False)
    bias: Optional[jnp.ndarray] = field(init=False, repr=False)
    counter: jnp.ndarray = field(init=False)
    w_init: Callable = field(default=jax.nn.initializers.normal(), repr=False)
    b_init: Callable = field(default=jax.nn.initializers.zeros, repr=False)
    parameters = pax.parameters_method("weight", "bias")

    def __post_init__(self):
        self.weight = self.w_init(pax.next_rng_key(), (self.in_dim, self.out_dim))
        self.bias = self.b_init(None, (self.out_dim,)) if self.with_bias else None
        self.counter = jnp.array(0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.counter += 1
        x = jnp.dot(x, self.weight)
        if self.with_bias:
            x = x + self.bias
        return x


pax.seed_rng_key(42)

fc = Linear(3, 4, name="fc1")

print("Before:", fc)
dummy_x = jnp.empty((32, 3))
fc, y = pax.purecall(fc, dummy_x)
assert y.shape == (32, 4)
print("After :", fc)
