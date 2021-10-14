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

        self.register_parameter("weight", jax.random.normal(pax.next_rng_key(), (1,)))
        self.register_parameter("bias", jax.random.normal(pax.next_rng_key(), (1,)))
        self.register_state("counter", jnp.array(0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        self.counter = self.counter + 1
        x = self.weight * x + self.bias
        return x


def loss_fn(model: Linear, x: jnp.ndarray, y: jnp.ndarray):
    model, y_hat = pax.module_and_value(model)(x)
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, (loss, model)


@jax.jit
def train_step(model: Linear, optimizer: GradientTransformation, x, y):
    grads, (loss, model) = pax.grad_parameters(loss_fn, has_aux=True)(model, x, y)
    updates, optimizer = opax.transform_gradients(grads, optimizer, model.parameters())
    new_params = opax.apply_updates(model.parameters(), updates)
    model = model.update_parameters(new_params)
    return model, optimizer, loss


net = Linear()
print(net.summary())
opt = opax.adam(1e-1)(net.parameters())


for step in range(10):
    net, opt, loss = train_step(net, opt, x, y)
    print(f"step {net.counter} loss {loss:.3f}")
