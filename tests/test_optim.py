import jax
import jax.numpy as jnp
import opax
import pax
import pytest


def test_optim_model_update_state():
    # a module updates it internal `count` value in the forward pass.

    class MyModule(pax.Module):
        count: int = 0
        fc: pax.Module

        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(2, 2)
            self.count = 0

        def __call__(self, x):
            self.count = self.count + 1
            x = self.fc(x)
            return x

    net = MyModule()

    def loss_fn(model: MyModule, x):
        y = model(x)
        loss = jnp.mean(jnp.square(x - y))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn=loss_fn)
    optimizer = opax.adamw()(net.parameters())
    x = jnp.zeros((2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError):
        net, optimizer, loss = update_fn(net, optimizer, x)


def test_sgd():
    class SGD(pax.Module):
        velocity: pax.Module
        learning_rate: float
        momentum: float

        def __init__(self, params, learning_rate: float = 1e-2, momentum: float = 0.9):
            super().__init__()
            self.momentum = momentum
            self.learning_rate = learning_rate
            self.velocity = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

        def step(self, grads: pax.Module, params: pax.Module):
            self.velocity = jax.tree_util.tree_map(
                lambda v, g: v * self.momentum + g * self.learning_rate,
                self.velocity,
                grads,
            )
            new_params = jax.tree_util.tree_map(
                lambda p, v: p - v, params, self.velocity
            )
            return new_params

    f = pax.Linear(2, 2)
    sgd = SGD(f, 0.9, 1e-4)
    pax.pure(sgd.step)(f, f)
