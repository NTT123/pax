from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import pax

pax.seed_rng_key(42)


@dataclass
class MLP(pax.AutoModule):
    features: Sequence[int]

    def __call__(self, x):
        sizes = zip(self.features[:-1], self.features[1:])
        for i, (in_dim, out_dim) in enumerate(sizes):
            fc = self.get_or_create(f"fc_{i}", lambda: pax.nn.Linear(in_dim, out_dim))
            x = jax.nn.relu(fc(x))
        return x


mlp, _ = MLP([1, 2, 3, 4, 5]) % jnp.ones((1, 1))
print(mlp.summary())
