import jax.numpy as jnp
import jmp
import pax


def test_mutate_new_module_list():
    a = pax.Linear(3, 3)
    b = a.copy()

    def mutate(b):
        b.lst = [pax.Linear(4, 4)]
        return b

    b = pax.pure(mutate)(b)
    # pylint: disable=protected-access
    # assert b.pax.name_to_kind["lst"] == pax.PaxKind.MODULE


def test_mp_policy_method_name():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.f = pax.Linear(3, 3)

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
    _ = m(x)  # ok

    _ = m.inference(x)
