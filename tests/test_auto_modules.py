import jax
import jax.numpy as jnp
import pax

# import pytest


def test_scan_bug_param_module():
    class M(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.register_state("a", jnp.array(0.0))

    # with pytest.raises(ValueError):
    _ = M()


def test_scan_bug_state_module():
    class M(pax.StateModule):
        def __init__(self):
            super().__init__()
            self.register_parameter("a", jnp.array(0.0))

    # with pytest.raises(ValueError):
    _ = M()


def test_auto_module():
    class M(pax.LazyModule):
        def __call__(self, x):
            x = self.get_or_create("fc", lambda: pax.nn.Linear(1, 1))(x)
            x = jax.nn.relu(x)
            return x

    m = M()
    x = jnp.ones((2, 1))
    m, _ = m % x
    print(m.summary())
