"""PAX and functional programming."""
import jax
import jax.numpy as jnp
import opax
import pax
from opax import GradientTransformation

pax.seed_rng_key(42)

# data
x = jax.random.normal(pax.next_rng_key(), (32, 1))
y = jax.random.normal(pax.next_rng_key(), (32, 1))


class Linear(pax.Module):

    weight: jnp.ndarray
    bias: jnp.ndarray
    counter: jnp.ndarray

    def __init__(self):
        super().__init__()

        with self.add_parameters():
            self.weight = jax.random.normal(pax.next_rng_key(), (1,))
            self.bias = jax.random.normal(pax.next_rng_key(), (1,))

        with self.add_states():
            self.counter = jnp.array(0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.counter = self.counter + 1
        x = self.weight * x + self.bias
        return x


def loss_fn(model: Linear, x: jnp.ndarray, y: jnp.ndarray):
    model, y_hat = model @ x
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, model


@jax.jit
def train_step(model: Linear, optimizer: GradientTransformation, x, y):
    grads, model, loss = pax.grad_mod_val(loss_fn)(model, x, y)
    optimizer, updates = optimizer @ (grads, ~model)
    model |= opax.apply_updates(~model, updates)
    return model, optimizer, loss


net = Linear()
print(net.summary())
opt = opax.adam(1e-1)(~net)


for step in range(10):
    net, opt, loss = train_step(net, opt, x, y)
    print(f"step {net.counter} loss {loss:.3f}")
