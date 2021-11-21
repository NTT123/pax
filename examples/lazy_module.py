"""A forward function that builds the model on the fly."""

import jax
import jax.numpy as jnp
import opax
import pax


@pax.pure
def forward(net: pax.experimental.LazyModule, x):
    fc1 = net.get_or_create("fc1", lambda: pax.Linear(1, 1))
    x = jax.nn.relu(fc1(x))
    fc2 = net.get_or_create("fc2", lambda: pax.Linear(1, 1))
    x = fc2(x)
    return net, x


def loss_fn(model, x: jnp.ndarray, y: jnp.ndarray):
    model, y_hat = forward(model, x)
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, model


@jax.jit
def train_step(model, optimizer: opax.GradientTransformation, x, y):
    (loss, model), grads = pax.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    model, optimizer = opax.apply_gradients(model, optimizer, grads)
    return model, optimizer, loss


def train():
    "train a lazy model."

    pax.seed_rng_key(42)

    # data
    x = jax.random.normal(pax.next_rng_key(), (32, 1))
    y = jax.random.normal(pax.next_rng_key(), (32, 1))

    # model & optimizer
    net, _ = forward(pax.experimental.LazyModule(), x)
    print(net.summary())
    opt = opax.adam(1e-1)(net.parameters())

    # training loop
    for step in range(10):
        net, opt, loss = train_step(net, opt, x, y)
        print(f"step {step} loss {loss:.3f}")

    return net


if __name__ == "__main__":
    train()
