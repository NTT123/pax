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
