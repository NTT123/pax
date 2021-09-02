import jax
import jax.numpy as jnp
import numpy as np
import pax
from pax.utils import LossFnOutput, RngSeq


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


def test_Rng_Seq():

    rng_seq = RngSeq(seed=42)
    assert rng_seq._rng_key.tolist() == [0, 42]
    r1 = rng_seq.next_rng_key()
    assert r1.shape == (2,)
    h1 = rng_seq._rng_key
    rs = rng_seq.next_rng_key(2)
    h2 = rng_seq._rng_key
    assert len(rs) == 2
    assert r1.tolist() != rs[0].tolist()
    assert h1.tolist() != h2.tolist(), "update internal state in `train` mode"

    rng_seq = rng_seq.eval()
    r3 = rng_seq.next_rng_key()
    r4 = rng_seq.next_rng_key()
    assert r3.tolist() == r4.tolist()
    h3 = rng_seq._rng_key
    assert h2.tolist() == h3.tolist(), "o update internal state in `eval` mode"
