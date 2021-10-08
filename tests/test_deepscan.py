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
    m._pax.name_to_kind["a"] == pax.PaxFieldKind.MODULE


@pax.pure
def test_assigned_field_an_array():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("a", np.array([3.0, 1.0], dtype=np.float32))

    # no error because we will automatically assign `a` to kind PARAMETER
    m = M()
    assert m._pax.name_to_kind["a"] == pax.PaxFieldKind.PARAMETER

    class N(pax.Module):
        def __init__(self):
            super().__init__()

    n = N()
    n = pax.scan_bugs(n)
    # no error because we will automatically assign `a` to kind PARAMETER
    n.register_parameter("b", jnp.array([1, 2, 3], dtype=jnp.float32))

    assert n._pax.name_to_kind["b"] == pax.PaxFieldKind.PARAMETER


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
        # m = pax.freeze_parameters(m)
        # d = OrderedDict(m._name_to_kind)
        # d["a"] = pax.module.PaxFieldKind.PARAMETER
        # m.__dict__["_name_to_kind"] = MappingProxyType(d)
        # m = pax.scan_bugs(m)


# def test_jit_():
#     class M(pax.Module):
#         def __init__(self):
#             super().__init__()
#             self.a_list = [pax.nn.Linear(2, 2)]

#         def __call__(self, x):
#             self.a_list.append(0)
#             return x

#     m = M()

#     @pax.jit_
#     def fwd(m, x):
#         return m(x)

#     with pytest.raises(ValueError):
#         x = fwd(m, jnp.zeros((2, 2)))
