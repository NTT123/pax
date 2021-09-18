from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import opax
import pax


def test_finetune():
    pax.seed_rng_key(42)

    class MLP(pax.Module):
        layers: List[pax.nn.Linear]

        def __init__(self, dims: List[int]):
            super().__init__()
            layers = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(pax.nn.Linear(in_dim, out_dim))
            self.layers = layers

        def __call__(self, x):
            for f in self.layers:
                x = f(x)
                x = jax.nn.sigmoid(x)
            return x

    net = MLP([10, 2, 2, 2, 10])

    def loss_fn(params: MLP, model: MLP, x):
        model = model.update(params)
        y = model(x)
        loss = jnp.mean(jnp.square(x - y))
        return loss, (loss, model)

    x = jax.random.normal(pax.next_rng_key(), (1, 10))

    # make all layers non-trainable except the last layer.
    for i in range(len(net.layers) - 1):
        net.layers[i] = net.layers[i].freeze()

    # net.layers[-1] = pax.nn.Linear(2, 10)
    optimizer = opax.adam(1e-2)(net.parameters())

    @pax.jit
    def update_fn(model: MLP, optimizer: pax.Module, x):
        params = model.parameters()
        grads, (loss, model) = pax.grad(loss_fn, has_aux=True)(params, model, x)
        model = model.update(
            optimizer.step(grads, model.parameters()),
        )
        return loss, model, optimizer

    old_layers = net.layers
    for i in range(100):
        loss, net, optimizer = update_fn(net, optimizer, x)
        if i % 10 == 0:
            print(f"[step {i:03d}] loss {loss:.3f}")
    new_layers = net.layers

    for i in range(len(net.layers) - 1):
        np.testing.assert_array_equal(old_layers[i].weight, new_layers[i].weight)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        old_layers[-1].weight,
        new_layers[-1].weight,
    )
