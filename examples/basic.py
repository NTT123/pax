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

    def __init__(self):
        super().__init__()

        self.register_parameter("weight", jax.random.normal(pax.next_rng_key(), (1,)))
        self.register_parameter("bias", jax.random.normal(pax.next_rng_key(), (1,)))

    def __call__(self, x):
        x = self.weight * x + self.bias
        return x


def loss_fn(model, x, y):
    y_hat = model(x)
    return jnp.mean(jnp.square(y_hat - y))


# jit with side effects
@pax.jit_
def train_step(model: Linear, optimizer: GradientTransformation, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    updates = optimizer(grads, model.parameters())
    new_params = pax.apply_updates(model.parameters(), updates=updates)
    model.update_parameters_(new_params)
    return loss


net = Linear()
print(net.summary())
opt = opax.adam(1e-1)(net.parameters())


for step in range(10):
    loss = train_step(net, opt, x, y)
    print(f"step {step} loss {loss:.3f}")
