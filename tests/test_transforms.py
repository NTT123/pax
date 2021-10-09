import jax.numpy as jnp
import jmp
import pax


def test_mutate_new_module_list():
    a = pax.nn.Linear(3, 3)

    b = a.copy()

    @pax.pure
    def _mutate(b):
        b.lst = []
        b.lst.append(pax.nn.Linear(4, 4))
        b.find_and_register_submodules()
        return b

    b = _mutate(b)

    assert b._pax.name_to_kind["lst"] == pax.PaxFieldKind.MODULE


def test_mp_policy_method_name():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.f = pax.nn.Linear(3, 3)

        def __call__(self, x):
            return self.f(x)

        def inference(self, x):
            return self.f(x) + 1.0

    m = M()
    half = jmp.half_dtype()
    full = jnp.float32

    p = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)

    m = pax.apply_mp_policy(m, mp_policy=p)
    x = jnp.zeros((4, 3))
    y = m(x)  # ok

    y = m.inference(x)
