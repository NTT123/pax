from typing import Tuple

import jax
import jax.numpy as jnp
import opax
import pax


def test_train_linear_regression_1():
    x = jax.random.normal(jax.random.PRNGKey(42), (32, 1), dtype=jnp.float32)
    noise = jax.random.normal(jax.random.PRNGKey(43), (32, 1), dtype=jnp.float32) * 0.2
    y = x * 2.5 - 3.1 + noise

    def loss_fn(model: pax.Linear, x, y):
        y_hat = model(x)
        loss = jnp.mean(jnp.square(y - y_hat))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    net = pax.Linear(1, 1)
    optimizer = opax.adamw(1e-1)(net.parameters())
    for step in range(100):
        net, optimizer, loss = update_fn(net, optimizer, x, y)
    print(f"[step {step}]  loss {loss:.3f}")


def test_train_linear_regression_2():
    x = jax.random.normal(jax.random.PRNGKey(42), (32, 1), dtype=jnp.float32)
    noise = jax.random.normal(jax.random.PRNGKey(43), (32, 1), dtype=jnp.float32) * 0.2
    y = x * 2.5 - 3.1 + noise

    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = pax.Linear(1, 32)
            self.fc2 = pax.Linear(32, 1)

        def __call__(self, x):
            x = self.fc1(x)
            x = jax.nn.relu(x)
            x = self.fc2(x)
            return x

    def loss_fn(model: M, x, y):
        y_hat = model(x)
        loss = jnp.mean(jnp.square(y - y_hat))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    net = M()
    optimizer = opax.adamw(1e-1)(net.parameters())
    for step in range(100):
        net, optimizer, loss = update_fn(net, optimizer, x, y)
    print(f"[step {step}]  loss {loss:.3f}")
