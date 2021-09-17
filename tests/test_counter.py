import jax
import jax.numpy as jnp
import pax


def test_counter():
    class Counter(pax.Module):
        counter: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, start_value: int = 0):
            super().__init__()
            self.counter = jnp.array(start_value, dtype=jnp.int32)
            self.bias = jnp.array(0.0)

        def __call__(self, x):
            self.counter = self.counter + 1
            return self.counter * x + self.bias

    def loss_fn(params: Counter, model: Counter, x: jnp.ndarray):
        model = model.update(params)
        y = model(x)
        loss = jnp.mean(jnp.square(x - y))
        return loss, (loss, model)

    grad_fn = jax.grad(loss_fn, has_aux=True)

    net = Counter(3)
    x = jnp.array(10.0)
    grads, (loss, net) = grad_fn(net.parameters(), net, x)
    assert grads.counter is None
    assert grads.bias.item() == 60.0
