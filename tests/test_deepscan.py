from types import MappingProxyType
from typing import OrderedDict

import jax.numpy as jnp
import numpy as np
import pax
import pytest


def test_list_of_mod():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.a = [pax.nn.Linear(3, 3)]

    m = M()
    m.get_kind("a") == pax.module.PaxFieldKind.MODULE_SUBTREE


def test_assigned_field_an_array():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("a", np.array([3.0, 1.0], dtype=np.float32))

    # no error because we will automatically assign `a` to kind PARAMETER
    m = M()
    assert m.get_kind("a") == pax.module.PaxFieldKind.PARAMETER

    class N(pax.Module):
        def __init__(self):
            super().__init__()

    n = N()
    n.deep_scan()
    # no error because we will automatically assign `a` to kind PARAMETER
    n.register_parameter("b", jnp.array([1, 2, 3], dtype=jnp.float32))
    assert n.get_kind("b") == pax.module.PaxFieldKind.PARAMETER


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
            self.a = np.array([3, 1], dtype=np.int32)

    with pytest.raises(ValueError):
        m = M()
        m = m.freeze()
        d = OrderedDict(m._name_to_kind)
        d["a"] = pax.module.PaxFieldKind.PARAMETER
        m.__dict__["_name_to_kind"] = MappingProxyType(d)
        m.deep_scan()
