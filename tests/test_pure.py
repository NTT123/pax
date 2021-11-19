import weakref
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pax
import pytest
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
    assert not jnp.array_equal(r1, r2)

    r3 = fn()
    _ = pax.next_rng_key()
    r4 = fn()
    assert_array_equal(r3, r4)


def test_deepcopy():
    class C(object):
        c: int

        def __init__(self):
            self.c = 0

    @pax.pure
    def mutate(x):
        x.c.c += 1
        return x

    class M(pax.Module):
        c: C

        def __init__(self):
            self.c = C()

    m = M()
    assert m.c.c == 0
    m1 = mutate(m)
    assert m.c.c == 1
    assert m1.c.c == 1


def test_deep_compare_1():
    class C(object):
        c: int

        def __init__(self):
            self.c = 0

    @pax.pure
    def mutate(x):
        return x

    class M(pax.Module):
        c: C

        def __init__(self):
            self.c = C()

    m = M()
    m1 = mutate(m)
    # with pytest.raises(AssertionError):
    pax.assert_structure_equal(m, m1)


def test_deep_compare_2():
    class C(object):
        c: int

        def __init__(self):
            self.c = 0

        def __eq__(self, o) -> bool:
            return self.c == o.c

    @pax.pure
    def mutate(x):
        return x

    class M(pax.Module):
        f: Any
        g: Any
        j: Any
        c: C

        def __init__(self):
            self.f = jax.nn.relu
            self.g = jax.nn.sigmoid
            self.j = partial(jax.nn.leaky_relu, negative_slope=0.2)
            self.h = jnp.tanh
            self.c = C()

    m = M()
    m1 = mutate(m)
    pax.assert_structure_equal(m, m1)


def test_module_weak_ref():
    mod = pax.nn.Linear(3, 3)
    mod_ref = weakref.ref(mod)
    assert mod_ref() is mod
    del mod
    assert mod_ref() is None


def test_abstraction_level_checking():
    def mutate(f):
        @jax.jit
        def g():
            f.a = "hello"

        g()

    fc = pax.nn.Linear(3, 3)
    with pytest.raises(ValueError):
        pax.pure(mutate)(fc)


def test_decorate_method_with_module_and_value():
    class M(pax.StateModule):
        def __init__(self):
            self.c = jnp.array(0)

        @pax.module_and_value
        def step(self):
            self.c += 1

    m = M()
    assert m.c.item() == 0
    m, _ = m.step()
    assert m.c.item() == 1
