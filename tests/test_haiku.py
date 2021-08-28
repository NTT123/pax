import jax
import jax.numpy as jnp
import numpy as np
import pax


def test_haiku_batchnorm():
    bn = pax.haiku.batch_norm_2d(32)
    x = np.ones((2, 10, 10, 32), dtype=np.float32)
    y = bn(x)
    assert y.shape == (2, 10, 10, 32)
    leaves, treedef = jax.tree_flatten(bn)
    bn = jax.tree_unflatten(treedef, leaves)


def test_haiku_batchnorm_grad():
    def fwd(params: pax.Module, model: pax.Module, x):
        loss = jnp.sum(model.update(params)(x))
        return loss

    bn = pax.haiku.batch_norm_2d(32)
    x = np.ones((2, 10, 10, 32), dtype=np.float32)
    grads = jax.grad(fwd, has_aux=False)(bn.parameters(), bn, x)
    leaves, treedef = jax.tree_flatten(grads)
    grads = jax.tree_unflatten(treedef, leaves)


def test_embed():
    embed = pax.haiku.embed(5, 32)
    x = np.zeros((2, 10), dtype=np.int32)
    y = embed(x)
    assert y.shape == (2, 10, 32)


def test_lstm():
    lstm = pax.haiku.lstm(32)
    x = np.zeros((2, 32))
    hx = lstm.initial_state(2)
    y, hx = lstm(x, hx)
    assert y.shape == (2, 32)
    assert hx.hidden.shape == (2, 32)
    assert hx.cell.shape == (2, 32)
