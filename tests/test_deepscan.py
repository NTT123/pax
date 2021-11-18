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
    # m.pax.name_to_kind["a"] == pax.PaxKind.MODULE


@pax.pure
def test_assigned_field_an_array():
    class M(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.a = np.array([3.0, 1.0], dtype=np.float32)

    # no error because we will automatically assign `a` to kind PARAMETER
    m = M()
    # assert m.pax.name_to_kind["a"] == pax.PaxKind.PARAMETER

    class N(pax.Module):
        def __init__(self):
            super().__init__()

    n = N()

    n.scan_bugs()
    # no error because we will automatically assign `a` to kind PARAMETER
    def mutate(n):
        n.b = jnp.array([1, 2, 3], dtype=jnp.float32)
        n.find_and_register_pytree_attributes()
        return n

    n = pax.pure(mutate)(n)

    # assert n.pax.name_to_kind["b"] == pax.PaxKind.PARAMETER


def test_assign_int_to_param():
    class M(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.a = np.array([3, 1], dtype=np.int32)

    _ = M()


def test_assign_int_to_param_deepscan():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.a = np.array([3, 1], dtype=np.int32)

    _ = M()
    # m = pax.freeze_parameters(m)
    # d = OrderedDict(m.name_to_kind)
    # d["a"] = pax.module.PaxKind.PARAMETER
    # m.__dict__["name_to_kind"] = MappingProxyType(d)
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
