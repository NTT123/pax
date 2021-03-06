import jax
import jax.numpy as jnp
import jmp
import pax
import pytest
from pax import apply_mp_policy

half = jmp.half_dtype()
full = jnp.float32


def test_wrap_unwrap_mp_policy():
    f = pax.Linear(3, 3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    ff = pax.apply_mp_policy(f, mp_policy=my_policy)
    fff = pax.unwrap_mp_policy(ff)
    assert hasattr(ff, "_pax_mp_policy")
    assert not hasattr(fff, "_pax_mp_policy")

    x = jax.numpy.ones((3, 3))
    assert f(x).dtype == full
    assert ff(x).dtype == half
    assert fff(x).dtype == full  # type: ignore


def test_sequential_mixed_precision():
    f = pax.Sequential(
        pax.Linear(3, 3),
        pax.BatchNorm2D(3, True, True, 0.9),
        pax.Linear(3, 3),
        pax.BatchNorm2D(3, True, True, 0.9),
    )
    linear_policy = jmp.Policy(compute_dtype=half, param_dtype=half, output_dtype=half)
    batchnorm_policy = jmp.Policy(
        compute_dtype=full, param_dtype=full, output_dtype=half
    )

    def policy_fn(mod):
        if isinstance(mod, pax.Linear):
            return pax.apply_mp_policy(mod, mp_policy=linear_policy)
        elif isinstance(mod, pax.BatchNorm2D):
            return pax.apply_mp_policy(mod, mp_policy=batchnorm_policy)
        else:
            # unchanged
            return mod

    f_mp = f.apply(policy_fn)
    x = jnp.zeros((32, 5, 5, 3))

    @pax.pure
    def run(f_mp):
        return f_mp(x)

    y = run(f_mp)
    assert y.dtype == half


def test_change_internal_state():
    class M(pax.Module):
        counter: jnp.ndarray

        def __init__(self):
            super().__init__()
            self.counter = jnp.array(0)

        def __call__(self, x):
            self.counter = self.counter + 1
            return x * self.counter

    m = M()
    mp = jmp.Policy(
        compute_dtype=jnp.float16, param_dtype=jnp.float32, output_dtype=jnp.float16
    )
    mm = m.apply(
        lambda x: (pax.apply_mp_policy(x, mp_policy=mp) if isinstance(x, M) else x)
    )
    x = jnp.array(0.0)
    assert mm.counter.item() == 0
    mm, y = pax.module_and_value(mm)(x)
    assert mm.counter.item() == 1
    assert m.counter.item() == 0


def test_change_tree_def():
    class M(pax.Module):
        counter: jnp.ndarray
        count: int

        def __init__(self):
            super().__init__()
            self.counter = jnp.array(0)
            self.count = 0

        def __call__(self, x):
            self.counter = self.counter + 1
            self.count = self.count + 1
            return x * self.counter

    m = M()
    mp = jmp.Policy(
        compute_dtype=jnp.float16, param_dtype=jnp.float32, output_dtype=jnp.float16
    )
    mm = m.apply(
        lambda x: (pax.apply_mp_policy(x, mp_policy=mp) if isinstance(x, M) else x)
    )
    x = jnp.array(0.0)
    assert mm.counter.item() == 0
    with pytest.raises(ValueError):
        y = mm(x)
    assert mm.counter.item() == 0
    assert m.counter.item() == 0


def test_wrap_wrap_mixed_precision():
    f = pax.Linear(3, 3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    f = pax.apply_mp_policy(f, mp_policy=my_policy)
    with pytest.raises(ValueError):
        f = pax.apply_mp_policy(f, mp_policy=my_policy)

    f = pax.unwrap_mp_policy(f)
    f = pax.apply_mp_policy(f, mp_policy=my_policy)

    with pytest.raises(ValueError):
        f = pax.apply_mp_policy(f, mp_policy=my_policy)


def test_mixed_precision_clone():
    f = pax.BatchNorm1D(3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    ff = pax.apply_mp_policy(f, mp_policy=my_policy)

    f = f.set_attribute("new_fc", pax.Linear(1, 1))
    # assert "new_fc" not in ff.pax.name_to_kind


def test_mixed_precision_unwrap_clone():
    f = pax.BatchNorm1D(3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    ff = pax.apply_mp_policy(f, mp_policy=my_policy)
    f = pax.unwrap_mp_policy(ff)
    f = f.set_attribute("new_fc", pax.Linear(1, 1))
    # assert "new_fc" not in ff.pax.name_to_kind


def test_mixed_precision_no_method_name():
    f = pax.Linear(3, 3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    # with pytest.raises(TypeError):
    _ = pax.apply_mp_policy(f, mp_policy=my_policy)


def test_mp_call_classmethod():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

        @classmethod
        def t(cls, y):
            return y

    m = M()
    x = jnp.zeros((3, 3))
    y = m.t(x)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    m = apply_mp_policy(m, mp_policy=my_policy)
    # with pytest.raises(ValueError):
    y = m.t(x)


def test_mp_call_staticmethod():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

        @staticmethod
        def t(_, y):
            return y

    m = M()
    x = jnp.zeros((3, 3))
    y = m.t(x, x)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    m = apply_mp_policy(m, mp_policy=my_policy)
    # with pytest.raises(ValueError):
    y = m.t(x, x)


@pax.pure
def test_mp_call_function():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    m = M()
    x = jnp.zeros((3, 3))

    def mutate(m):
        m.q = lambda x: x
        return m

    m = pax.pure(mutate)(m)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)
    m = apply_mp_policy(m, mp_policy=my_policy)
    # with pytest.raises(ValueError):
    m.q(x)
