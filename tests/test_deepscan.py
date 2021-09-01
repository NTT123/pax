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
    n.deep_scan()
    with pytest.raises(ValueError):
        n.b = jnp.array([1, 2, 3])


def test_assign_int_to_param():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("a", np.array([3, 1], dtype=np.int32))

    with pytest.raises(ValueError):
        m = M()


def test_assign_int_to_param_deepscan():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_state("a", np.array([3, 1], dtype=np.int32))

    m = M()
    m = m.freeze()
    m._name_to_kind["a"] = pax.module.PaxFieldKind.PARAMETER
    with pytest.raises(ValueError):
        m.deep_scan()
