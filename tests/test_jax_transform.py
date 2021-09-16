import jax.numpy as jnp
import pax
import pytest


def test_jit_immutability():
    class M(pax.Module):
        def __init__(self):
            self.x = pax.nn.Linear(2, 2)
            self.counter = 2

        def __call__(self, x):
            print("call")
            self.counter = self.counter + 1
            return x

    m = M()
    x = jnp.zeros((1, 1))
    with pytest.raises(ValueError):
        y = pax.jit(lambda y: m(y))(x)
