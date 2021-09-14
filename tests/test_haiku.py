import haiku as hk
import jax
import numpy as np
import pax


def test_haiku_conv1d_online():
    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_conv1d_online_jit():
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
    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)

    assert f._is_haiku_initialized == False
    f = f.hk_init(x)
    assert f._is_haiku_initialized == True
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_haiku_conv1d_hk_init_jit():
    f = pax.from_haiku(hk.Conv1D)(4, 3, padding="VALID")
    x = np.zeros((3, 5, 3), dtype=np.float32)

    assert f._is_haiku_initialized == False
    f = f.hk_init(x, enable_jit=True)
    assert f._is_haiku_initialized == True
    y = f(x)
    assert y.shape == (3, 3, 4)


def test_from_haiku_linear():
    x = np.zeros((2, 1), dtype=np.float32)
    f = pax.from_haiku(hk.Linear)(32)
    y = f(x)
    assert y.shape == (2, 32)
