import jax
import jax.numpy as jnp
import pax
from pax.transforms import select_parameters


def test_counter():
    class Counter(pax.Module):
        counter: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, start_value: int = 0):
            super().__init__()
            self.register_state("counter", jnp.array(start_value, dtype=jnp.int32))
            self.register_parameter("bias", jnp.array(0.0))

        def __call__(self, x):
            self.counter = self.counter + 1
            return self.counter * x + self.bias

    def loss_fn(params: Counter, model: Counter, x: jnp.ndarray):
        model = pax.update_parameters(model, params=params)
        y = model(x)
        loss = jnp.mean(jnp.square(x - y))
        return loss, (loss, model)

    grad_fn = jax.grad(loss_fn, has_aux=True)

    net = Counter(3)
    x = jnp.array(10.0)
    grads, (loss, net) = grad_fn(select_parameters(net), net, x)
    assert grads.counter is None
    assert grads.bias.item() == 60.0
