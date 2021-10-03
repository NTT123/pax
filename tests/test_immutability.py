import jax.numpy as jnp
import pax
import pytest


def test_immutability():
    f = pax.nn.Linear(3, 3)

    with pytest.raises(ValueError):
        f.c = 123

    g = pax.freeze_parameters(f)
    k = pax.unfreeze_parameters(g, origin=f)


def test_new_empty_attribute():
    class M(pax.Module):
        a = []

    m = M()


def test_new_unregistered_array():
    class M(pax.Module):
        a = [jnp.zeros((3, 3))]

    with pytest.raises(ValueError):
        m = M()


def test_new_unregistered_module():
    class M(pax.Module):
        a = pax.nn.Linear(3, 3)

    with pytest.raises(ValueError):
        m = M()
