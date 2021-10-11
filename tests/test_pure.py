import jax
import pax
from numpy.testing import assert_array_equal


def test_rng_unchanged():
    pax.seed_rng_key(41)
    pax.next_rng_key()

    @jax.jit
    @pax.pure
    def fn():
        return pax.next_rng_key()

    def f1():
        pax.seed_rng_key(41)
        pax.next_rng_key()
        return pax.next_rng_key()

    def f2():
        pax.seed_rng_key(41)
        pax.next_rng_key()
        fn()
        return pax.next_rng_key()

    r1 = f1()
    r2 = f2()
    assert_array_equal(r1, r2)

    r3 = fn()
    _ = pax.next_rng_key()
    r4 = fn()
    assert_array_equal(r3, r4)
