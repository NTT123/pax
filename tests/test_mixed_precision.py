import jax
import jax.numpy as jnp
import jmp
import pax
from pax.nn.batch_norm import BatchNorm2D
from pax.nn.linear import Linear

half = jnp.float16  # On TPU this should be jnp.bfloat16.
full = jnp.float32


def test_wrap_unwrap_mixed_precision():
    f = pax.nn.Linear(3, 3)
    my_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=half)

    ff = f.mixed_precision(my_policy)
    fff = ff.unwrap_mixed_precision()
    print(f.__class__.__name__)
    print(ff.__class__.__name__)
    print(fff.__class__.__name__)

    x = jax.numpy.ones((3, 3))
    print("f", f(x))
    print("ff", ff(x))
    print("fff", fff(x))


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
    print(f_mp.summary())
    x = jnp.zeros((32, 5, 5, 3))
    y = f_mp(x)
    print(y.shape, y.dtype)
