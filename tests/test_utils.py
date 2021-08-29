import jax
import jax.numpy as jnp
import numpy as np
import pax
from pax.utils import LossFnOutput


def test_util_update_fn():
    def loss_fn(params: pax.nn.Linear, model: pax.nn.Linear, inputs) -> LossFnOutput:
        x, target = inputs
        model = model.update(params)
        y = model(x)
        loss = jnp.mean(jnp.square(y - target))
        return loss, (loss, model)

    net = pax.nn.Linear(2, 1)
    opt = pax.optim.adamw(net.parameters(), learning_rate=1e-1)
    update_fn = jax.jit(pax.utils.build_update_fn(loss_fn))
    x = np.random.normal(size=(32, 2))
    y = np.random.normal(size=(32, 1))
    for step in range(3):
        loss, net, opt = update_fn(net, opt, (x, y))
    print(step, loss)
