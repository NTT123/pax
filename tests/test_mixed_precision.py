import jax
import jax.numpy as jnp
import jmp
import pax
import pytest

half = jnp.float16  # On TPU this should be jnp.bfloat16.
full = jnp.float32


def test_wrap_unwrap_mixed_precision():
    f = pax.nn.Linear(3, 3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    ff = f.mixed_precision(my_policy)
    fff = ff.unwrap_mixed_precision()
    assert "MixedPrecision" in ff.__class__.__name__
    assert "MixedPrecision" not in fff.__class__.__name__

    x = jax.numpy.ones((3, 3))
    assert f(x).dtype == full
    assert ff(x).dtype == half
    assert fff(x).dtype == full


def test_sequential_mixed_precision():
    f = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.BatchNorm2D(3, True, True, 0.9),
        pax.nn.Linear(3, 3),
        pax.nn.BatchNorm2D(3, True, True, 0.9),
    )
    linear_policy = jmp.Policy(compute_dtype=half, param_dtype=half, output_dtype=half)
    batchnorm_policy = jmp.Policy(
        compute_dtype=full, param_dtype=full, output_dtype=half
    )

    def policy_fn(mod):
        if isinstance(mod, pax.nn.Linear):
            return mod.mixed_precision(linear_policy)
        elif isinstance(mod, pax.nn.BatchNorm2D):
            return mod.mixed_precision(batchnorm_policy)
        else:
            # unchanged
            return mod

    f_mp = f.apply(policy_fn)
    x = jnp.zeros((32, 5, 5, 3))
    y = f_mp(x)
    assert y.dtype == half


def test_change_internal_state():
    class M(pax.Module):
        counter: jnp.ndarray

        def __init__(self):
            super().__init__()
            self.register_state("counter", jnp.array(0))

        def __call__(self, x):
            self.counter = self.counter + 1
            return x * self.counter

    m = M()
    mp = jmp.Policy(
        compute_dtype=jnp.float16, param_dtype=jnp.float32, output_dtype=jnp.float16
    )
    mm = m.apply(lambda x: (x.mixed_precision(mp) if isinstance(x, M) else x))
    x = jnp.array(0.0)
    assert mm.counter.item() == 0
    y = mm(x)
    assert mm.counter.item() == 1
    assert m.counter.item() == 0


def test_change_tree_def():
    class M(pax.Module):
        counter: jnp.ndarray
        count: int

        def __init__(self):
            super().__init__()
            self.register_state("counter", jnp.array(0))
            self.count = 0

        def __call__(self, x):
            self.counter = self.counter + 1
            self.count = self.count + 1
            return x * self.counter

    m = M()
    mp = jmp.Policy(
        compute_dtype=jnp.float16, param_dtype=jnp.float32, output_dtype=jnp.float16
    )
    mm = m.apply(lambda x: (x.mixed_precision(mp) if isinstance(x, M) else x))
    x = jnp.array(0.0)
    assert mm.counter.item() == 0
    with pytest.raises(RuntimeError):
        y = mm(x)
    assert mm.counter.item() == 1
    assert m.counter.item() == 0
