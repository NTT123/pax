import jax.numpy as jnp
import pax
import pytest


def test_jit_immutability():
    class M(pax.Module):
        def __init__(self):
            self.x = pax.nn.Linear(2, 2)
            self.counter = 2

        def __call__(self, x):
            self.counter = self.counter + 1
            return x

    m = M()
    x = jnp.zeros((1, 1))
    with pytest.raises(ValueError):
        y = pax.jit(lambda y: m(y))(x)


def test_grad_deepscan():
    class M(pax.Module):
        def __init__(self):
            self.fc = pax.nn.Linear(2, 2)

        def __call__(self, x):
            return self.fc(x)

    def loss_fn(params, model, inputs):
        model = pax.update_parameters(model, params=params)
        loss = jnp.mean(model(inputs))
        return loss, (loss, model)

    m = M()
    x = jnp.zeros((1, 2))
    m.__dict__["fc1"] = pax.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        y = pax.grad(loss_fn, has_aux=True)(pax.select_parameters(m), m, x)


def test_loss_fn_no_return_model():
    def loss_fn(params, model, inputs):
        model = pax.update_parameters(model, params=params)
        y = model(inputs)
        return jnp.sum(y)

    grad_fn = pax.grad(loss_fn)
    x = jnp.zeros((3, 3))
    net = pax.nn.Linear(3, 3)
    with pytest.raises(ValueError):
        y = grad_fn(net.parameters(), net, x)


def test_jit__call__():
    class M(pax.Module):
        @pax.jit
        def __call__(self, x):
            return x, self

    x = jnp.zeros((3, 3))
    net = M()
    y = net(x)

    class M(pax.Module):
        @pax.jit
        def __call__(self, x):
            return x

    net = M()
    with pytest.raises(ValueError):
        y = net(x)
