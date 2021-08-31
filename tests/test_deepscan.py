import jax.numpy as jnp
import numpy as np
import pax
import pytest


def test_list_of_mod():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.a = [pax.nn.Linear(3, 3)]

    with pytest.raises(ValueError):
        m = M()


def test_assigned_field_an_array():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.a = np.array([3.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError):
        m = M()

    class N(pax.Module):
        def __init__(self):
            super().__init__()

    n = N()
    with pytest.raises(ValueError):
        n.b = jnp.array([1, 2, 3])


def test_assign_int_to_param():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("a", np.array([3, 1], dtype=np.int32))

    with pytest.raises(ValueError):
        m = M()
