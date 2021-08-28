from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pax
from pax.tree import Leaf


def test_finetune():
    pax.seed_rng_key(42)

    class MLP(pax.Module):
        layers: List[pax.Module]

        def __init__(self, dims: List[int]):
            self.layers = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                self.layers.append(pax.nn.Linear(in_dim, out_dim))

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

    def freeze_filter(node: Leaf, model: MLP):
        # freeze all layers except the last one.
        freezed_layers = model.layers[0:-1]
        return node.info["module"] not in freezed_layers

    net = net.filter_parameters(freeze_filter)
    optimizer = pax.optim.from_optax(optax.adam(1e-2))(net.parameters())

    @jax.jit
    def update_fn(model: MLP, optimizer: pax.Optimizer, x):
        params = model.parameters()
        grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(params, model, x)
        model = optimizer.step(grads, model)
        return loss, model, optimizer

    old_layers = net.layers
    for i in range(100):
        loss, net, optimizer = update_fn(net, optimizer, x)
        if i % 10 == 0:
            print(f"[step {i:03d}] loss {loss:.3f}")
    new_layers = net.layers

    for i in range(len(net.layers) - 1):
        np.testing.assert_array_equal(old_layers[i].W, new_layers[i].W)

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        old_layers[-1].W,
        new_layers[-1].W,
    )
