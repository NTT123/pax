import jax.numpy as jnp
import pax
import pytest


def test_adamw_optim():
    net = pax.nn.Linear(1, 1)
    opt = pax.optim.adamw(net.parameters())
    net = opt.step(net.parameters(), net)


def test_optim_model_update_state():
    # a module updates it internal `count` value in the forward pass.

    class MyModule(pax.Module):
        count: int = 0
        fc: pax.Module

        def __init__(self):
            self.fc = pax.nn.Linear(2, 2)
            self.count = 0

        def __call__(self, x):
            self.count = self.count + 1
            x = self.fc(x)
            return x

    net = MyModule()

    def loss_fn(params: MyModule, model: MyModule, x):
        model = model.update(params)
        y = model(x)
        loss = jnp.mean(jnp.square(x - y))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn=loss_fn)
    optimizer = pax.optim.adamw(net.parameters())
    x = jnp.zeros((2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError):
        loss, net, optimizer = update_fn(net, optimizer, x)
