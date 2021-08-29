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
    x = np.zeros((2, 32), dtype=np.float32)
    hx = lstm.initial_state(2)
    y, hx = lstm(x, hx)
    assert y.shape == (2, 32)
    assert hx.hidden.shape == (2, 32)
    assert hx.cell.shape == (2, 32)


def test_haiku_gru():
    lstm = pax.haiku.gru(32)
    x = np.zeros((2, 32), dtype=np.float32)
    hx = lstm.initial_state(2)
    y, hx = lstm(x, hx)
    assert y.shape == (2, 32)
    assert hx.shape == (2, 32)


def test_haiku_conv_1d():
    conv1d = pax.haiku.conv_1d(3, 4, 2)
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = conv1d(x)
    assert y.shape == (3, 5, 4)


def test_haiku_conv_2d():
    conv = pax.haiku.conv_2d(3, 4, 2)
    x = np.zeros((3, 5, 6, 3), dtype=np.float32)
    y = conv(x)
    assert y.shape == (3, 5, 6, 4)


def test_haiku_conv1d_transpose():
    conv_transpose = pax.haiku.conv_1d_transpose(3, 4, 2)
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = conv_transpose(x)
    assert y.shape == (3, 5, 4)


def test_haiku_conv2d_transpose():
    conv_transpose = pax.haiku.conv_2d_transpose(3, 4, 2)
    x = np.zeros((3, 5, 6, 3), dtype=np.float32)
    y = conv_transpose(x)
    assert y.shape == (3, 5, 6, 4)


def test_haiku_conv1d_online():
    import haiku as hk

    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_conv1d_online_jit():
    import haiku as hk

    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)

    @jax.jit
    def init(x, model):
        model(x)
        return model

    assert f._is_haiku_initialized == False
    f = init(x, f)
    assert f._is_haiku_initialized == True
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_conv1d_hk_init():
    import haiku as hk

    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)

    assert f._is_haiku_initialized == False
    f = f.hk_init(x)
    assert f._is_haiku_initialized == True
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_conv1d_hk_init_jit():
    import haiku as hk

    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)

    assert f._is_haiku_initialized == False
    f = f.hk_init(x, enable_jit=True)
    assert f._is_haiku_initialized == True
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_avg_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    avg_pool = pax.haiku.avg_pool(2, 2, "SAME", -1)
    y = avg_pool(x)
    assert y.shape == (3, 3, 3)


def test_haiku_max_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    avg_pool = pax.haiku.avg_pool(2, 3, "SAME", -1)
    y = avg_pool(x)
    assert y.shape == (3, 2, 3)


def test_from_haiku_linear():
    x = np.zeros((2, 1), dtype=np.float32)
    import haiku as hk

    f = pax.from_haiku(hk.Linear)(32)
    y = f(x)
    assert y.shape == (2, 32)
