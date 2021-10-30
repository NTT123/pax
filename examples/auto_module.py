from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import opax
import pax

pax.seed_rng_key(42)


@dataclass
class MLP(pax.AutoModule):
    features: Sequence[int]

    def __call__(self, x):
        sizes = zip(self.features[:-1], self.features[1:])
        for i, (in_dim, out_dim) in enumerate(sizes):
            create_fn = lambda: pax.nn.Linear(in_dim, out_dim)
            x = self.get_or_create(f"fc_{i}", create_fn)(x)
            x = jax.nn.leaky_relu(x)
        return x

    def __post_init__(self):
        # initialize submodules with a test run
        self(jnp.empty((1, self.features[0])))


def loss_fn(net: MLP, x, y):
    net, y_hat = net % x
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, net


@jax.jit
def update_fn(net, optimizer: opax.GradientTransformation, x, y):
    (loss, net), grads = pax.value_and_grad(loss_fn)(net, x, y)
    net, optimizer = opax.apply_gradients(net, optimizer, grads)
    return net, optimizer, loss


net = MLP([1, 2, 3, 4, 5])
optimizer = opax.adam(1e-2)(~net)

print(net.summary())
print(optimizer)

x = jnp.ones((32, 1))
y = jnp.ones((32, 5))

for step in range(100):
    net, optimizer, loss = update_fn(net, optimizer, x, y)
    if step % 10 == 0:
        print(f"step {step:03d}  loss {loss:.3f}")
