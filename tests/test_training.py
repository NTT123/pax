from typing import Tuple

import jax
import jax.numpy as jnp
import pax
from pax.utils import LossFnOutput


def test_train_linear_regression():
    x = jax.random.normal(jax.random.PRNGKey(42), (32, 1), dtype=jnp.float32)
    noise = jax.random.normal(jax.random.PRNGKey(43), (32, 1), dtype=jnp.float32) * 0.2
    y = x * 2.5 - 3.1 + noise

    def loss_fn(
        params: pax.nn.Linear,
        model: pax.nn.Linear,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> LossFnOutput:
        model = model.update(params)
        x, y = inputs
        y_hat = model(x)
        loss = jnp.mean(jnp.square(y - y_hat))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    net = pax.nn.Linear(1, 1)
    optimizer = pax.optim.adamw(net.parameters(), 1e-1)
    for step in range(100):
        loss, net, optimizer = update_fn(net, optimizer, (x, y))
    print(f"[step {step}]  loss {loss:.3f}")


def test_train_hk_linear_regression():
    x = jax.random.normal(jax.random.PRNGKey(42), (32, 1), dtype=jnp.float32)
    noise = jax.random.normal(jax.random.PRNGKey(43), (32, 1), dtype=jnp.float32) * 0.2
    y = x * 2.5 - 3.1 + noise

    import haiku as hk

    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = pax.from_haiku(hk.Linear)(32)
            self.fc2 = pax.from_haiku(hk.Linear)(1)

        def __call__(self, x):
            x = self.fc1(x)
            x = jax.nn.relu(x)
            x = self.fc2(x)
            return x

    def loss_fn(
        params: M,
        model: M,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> LossFnOutput:
        model = model.update(params)
        x, y = inputs
        y_hat = model(x)
        loss = jnp.mean(jnp.square(y - y_hat))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    net = M().hk_init(x)
    optimizer = pax.optim.adamw(net.parameters(), 1e-1)
    for step in range(100):
        loss, net, optimizer = update_fn(net, optimizer, (x, y))
    print(f"[step {step}]  loss {loss:.3f}")
