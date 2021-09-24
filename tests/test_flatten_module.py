import jax
import pax
import pytest


def test_flatten_module():
    f = pax.nn.Linear(4, 4)
    g = pax.nn.FlattenModule(f)
    k = pax.select_parameter(g)
    assert jax.tree_structure(g) == jax.tree_structure(k)
    h = g.update(k)


def test_none_state():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_state_subtree("s", [])

    m = M()
    p = pax.select_parameter(m)
    assert p.s == []
    pax.utils.assertStructureEqual(m, p)


def test_flatten_non_callable_module():
    class M(pax.Module):
        def __init__(self):
            super().__init__()

    m = M()

    with pytest.raises(ValueError):
        n = pax.nn.FlattenModule(m)


def test_flatten_module_freeze():
    a = pax.nn.Linear(1, 1)
    b = pax.nn.FlattenModule(a)

    with pytest.raises(RuntimeError):
        c = b.freeze()


def test_flatten_module_unfreeze():
    a = pax.nn.Linear(1, 1)
    b = pax.nn.FlattenModule(a)

    with pytest.raises(RuntimeError):
        c = b.unfreeze()


def test_flatten_module_eval():
    a = pax.nn.Linear(1, 1)
    b = pax.nn.FlattenModule(a)

    # with pytest.raises(RuntimeError):
    c = pax.enable_eval_mode(b)
