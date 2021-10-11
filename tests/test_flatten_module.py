from typing import List

import jax
import pax


def test_flatten_module():
    f = pax.nn.Linear(4, 4)
    g = pax.flatten_module(f)
    k = pax.select_parameters(g)
    assert jax.tree_structure(g) == jax.tree_structure(k)
    h = g.update_parameters(k)


def test_none_state():
    class M(pax.Module):
        s: List

        def __init__(self):
            super().__init__()
            self.register_states("s", [])

    m = M()
    p = pax.select_parameters(m)
    assert p.s == []
    pax.assertStructureEqual(m, p)


def test_flatten_non_callable_module():
    class M(pax.Module):
        def __init__(self):
            super().__init__()

        def __call__(self, x):
            return x

    m = M()

    # with pytest.raises(ValueError):
    n = pax.flatten_module(m)
    k = n.unflatten()
    assert type(k) == M


def test_flatten_module_freeze():
    a = pax.nn.Linear(1, 1)
    b = pax.flatten_module(a)

    # with pytest.raises(RuntimeError):
    c = pax.freeze_parameters(b)


def test_flatten_module_unfreeze():
    a = pax.nn.Linear(1, 1)
    b = pax.flatten_module(a)

    # with pytest.raises(RuntimeError):
    # c = b.unfreeze()


def test_flatten_module_eval():
    a = pax.nn.Linear(1, 1)
    b = pax.flatten_module(a)

    # with pytest.raises(RuntimeError):
    c = pax.enable_eval_mode(b)
