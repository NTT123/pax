import jax
import numpy as np
import pax


def test_embed():
    embed = pax.nn.Embed(5, 32)
    x = np.zeros((2, 10), dtype=np.int32)
    y = embed(x)
    assert y.shape == (2, 10, 32)


def test_lstm():
    lstm = pax.nn.LSTM(32, 32)
    x = np.zeros((2, 32), dtype=np.float32)
    hx = lstm.initial_state(2)
    hx, y = lstm(hx, x)
    assert y.shape == (2, 32)
    assert hx.hidden.shape == (2, 32)
    assert hx.cell.shape == (2, 32)


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


def test_avg_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = pax.nn.avg_pool(x, (2, 1), (2, 1), "SAME", -1)
    assert y.shape == (3, 3, 3)


def test_haiku_max_pool():
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = pax.nn.max_pool(x, (2, 1), (3, 1), "SAME", -1)
    assert y.shape == (3, 2, 3)


def test_from_haiku_linear():
    x = np.zeros((2, 1), dtype=np.float32)
    import haiku as hk

    f = pax.from_haiku(hk.Linear)(32)
    y = f(x)
    assert y.shape == (2, 32)
