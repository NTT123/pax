import jax.numpy as jnp
import numpy as np
import opax
import pax
from pax import LossFnOutput
from pax.nn import EMA, RngSeq


def test_util_update_fn():
    def loss_fn(params: pax.nn.Linear, model: pax.nn.Linear, inputs) -> LossFnOutput:
        x, target = inputs
        model = pax.update_params(model, params=params)
        y = model(x)
        loss = jnp.mean(jnp.square(y - target))
        return loss, (loss, model)

    net = pax.nn.Linear(2, 1)
    opt = opax.adamw(learning_rate=1e-1)(net.parameters())
    update_fn = pax.jit(pax.utils.build_update_fn(loss_fn))
    x = np.random.normal(size=(32, 2))
    y = np.random.normal(size=(32, 1))
    for step in range(3):
        (net, opt), loss = update_fn((net, opt), (x, y))
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

    rng_seq = pax.enable_eval_mode(rng_seq)
    r3 = rng_seq.next_rng_key()
    r4 = rng_seq.next_rng_key()
    assert r3.tolist() == r4.tolist()
    h3 = rng_seq._rng_key
    assert h2.tolist() == h3.tolist(), "o update internal state in `eval` mode"


def test_ema_debias():
    ema = EMA(jnp.array(1.0), 0.9, True)
    assert ema.debias.item() == False
    assert ema.averages.item() == 1.0
    ema(jnp.array(2.0))
    assert ema.averages.item() == 2.0
    assert ema.debias.item() == True
    ema(jnp.array(1.0))
    np.testing.assert_almost_equal(ema.averages.item(), 0.9 * 2.0 + 0.1 * 1.0)


def test_ema_bias():
    ema = EMA(jnp.array(1.0), 0.9, False)
    assert ema.debias is None
    assert ema.averages.item() == 1.0
    ema(jnp.array(2.0))
    np.testing.assert_almost_equal(ema.averages.item(), 0.1 * 2.0 + 0.9 * 1.0)
