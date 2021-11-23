"""PAX basic stuffs."""

import jax
import jax.numpy as jnp
import opax
import pax
from opax import GradientTransformation


class Linear(pax.Module):
    """A linear module with counter."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    counter: jnp.ndarray
    parameters = pax.parameters_method("weight", "bias")

    def __init__(self):
        super().__init__()
        self.weight = jax.random.normal(pax.next_rng_key(), (1,))
        self.bias = jax.random.normal(pax.next_rng_key(), (1,))
        self.counter = jnp.array(0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.counter = self.counter + 1
        x = self.weight * x + self.bias
        return x


def loss_fn(model: Linear, x: jnp.ndarray, y: jnp.ndarray):
    model, y_hat = pax.module_and_value(model)(x)
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, model


@jax.jit
def train_step(model: Linear, optimizer: GradientTransformation, x, y):
    (loss, model), grads = pax.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    model, optimizer = opax.apply_gradients(model, optimizer, grads)
    return model, optimizer, loss


def main():
    # random seed
    pax.seed_rng_key(42)

    # model & optimizer
    net = Linear()
    print(net.summary())
    opt = opax.adam(1e-1).init(net.parameters())

    # data
    x = jax.random.normal(pax.next_rng_key(), (32, 1))
    y = jax.random.normal(pax.next_rng_key(), (32, 1))

    # training loop
    for _ in range(10):
        net, opt, loss = train_step(net, opt, x, y)
        print(f"step {net.counter:>2} loss {loss:.3f}")


if __name__ == "__main__":
    main()
