"""Test important pax stuffs."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pax
import pytest
from pax import Module


def test_pax_next_rng_key():
    # seed 42
    pax.seed_rng_key(42)
    # assert pax.rng.state._rng_key is None
    # assert pax.rng.state._seed == 42
    expected_rng = jnp.array([0, 42], dtype=jnp.uint32)
    rng1 = pax.next_rng_key()
    expected_rng_1, rng_internal = jax.random.split(expected_rng)
    np.testing.assert_array_equal(rng1, expected_rng_1)
    rng2 = pax.next_rng_key()
    expected_rng_2, rng_internal = jax.random.split(rng_internal)
    np.testing.assert_array_equal(rng2, expected_rng_2)


def test_type_union():
    class Counter(pax.Module):
        count: Union[int, jnp.ndarray]

        def __init__(self):
            super().__init__()
            self.register_state("count", [0])

    with pytest.raises(ValueError):
        counter = Counter()
        leaves, treedef = jax.tree_flatten(counter)
        assert leaves == [0]


def test_type_list_int():
    class Counter(pax.Module):
        count: List[int]

        def __init__(self):
            super().__init__()
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_sequence():
    class Counter(pax.Module):
        count: Sequence[int]

        def __init__(self):
            super().__init__()
            self.register_parameter("count", jnp.array([0.0]))

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [jnp.array(0.0)]


def test_type_dict():
    class Counter(pax.Module):
        count: Dict[str, int]

        def __init__(self):
            super().__init__()
            self.register_states("count", {"conv1": [1, 2, 3], "conv2": ["a", "b"]})

    with pytest.raises(ValueError):
        counter = Counter()
        leaves, treedef = jax.tree_flatten(counter)
        assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, int]]

        def __init__(self):
            super().__init__()
            self.register_state(
                "count", {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}
            )

    with pytest.raises(ValueError):
        counter = Counter()
        leaves, treedef = jax.tree_flatten(counter)
        assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict_optional():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, Optional[int]]]

        def __init__(self):
            super().__init__()
            self.register_state(
                "count", {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}
            )

    with pytest.raises(ValueError):
        counter = Counter()
        leaves, treedef = jax.tree_flatten(counter)
        assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict_optional1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, List[Union[int, str]]]]

        def __init__(self):
            super().__init__()
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_tuple():
    class Counter(pax.Module):
        count: Tuple[int, int]

        def __init__(self):
            super().__init__()
            self.count = (1, 2)

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_optional():
    class Counter(pax.Module):
        count: Optional[int]

        def __init__(self):
            super().__init__()
            self.register_state("count", jnp.array(0))

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_train_eval():
    net = pax.nn.Sequential(pax.nn.Linear(3, 3), pax.nn.Linear(3, 3))

    assert net.is_training() == True
    net = pax.enable_eval_mode(net)
    assert net.is_training() == False
    assert net.modules[0].is_training() == False
    assert net.modules[1].is_training() == False
    net = pax.enable_train_mode(net)
    assert net.is_training() == True
    assert net.modules[0].is_training() == True
    assert net.modules[1].is_training() == True


def test_state_of_param():
    class M1(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("p1", jnp.array(0.0, dtype=jnp.float32))

    m1 = M1()

    class M2(pax.Module):
        def __init__(self, m11):
            super().__init__()
            self.register_states("m2", {"m1": m11})

    m2 = M2(m1)
    assert len(jax.tree_leaves(pax.select_kind(m1, kind=pax.PaxFieldKind.STATE))) == 0
    assert (
        len(jax.tree_leaves(pax.select_kind(m2, kind=pax.PaxFieldKind.PARAMETER))) == 0
    )

    assert (
        len(jax.tree_leaves(pax.select_kind(m1, kind=pax.PaxFieldKind.PARAMETER))) == 1
    )
    assert len(jax.tree_leaves(pax.select_kind(m2, kind=pax.PaxFieldKind.STATE))) == 1


def test_module_properties_modify():
    fc = pax.nn.Linear(3, 3)
    assert fc.is_training() == True
    fc1 = fc.copy()
    assert fc1.is_training() == True
    fc = pax.enable_eval_mode(fc)
    assert fc.is_training() == False
    assert fc1.is_training() == True


def test_clone_no_side_effect():
    fc1 = pax.nn.Linear(3, 3)
    fc2 = fc1.copy()

    fc1.new_module = pax.nn.Linear(5, 5)

    assert (
        "new_module" in fc1._pax.name_to_kind
    ), "registered 'new_modules' as part of fc1"
    assert (
        "new_module" not in fc2._pax.name_to_kind
    ), "fc2._pax.name_to_kind is different from fc1._pax.name_to_kind"


def test_lambda_module():
    f = pax.nn.Lambda(jax.nn.relu)
    x = jnp.array(5.0)
    y = f(x)
    assert x.item() == y.item()

    x = jnp.array(-4.0)
    y = f(x)
    assert y.item() == 0.0


def test_forget_call_super_at_init():
    class M(pax.Module):
        def __init__(self):
            self.fc = pax.nn.Linear(3, 3)

    # with initialization in the `__new__` method,
    # no need to call `super().__init__()` anymore.
    m = M()


def test_name_repr():
    fc = pax.nn.Linear(2, 3, name="fc1")
    assert "(fc1)" in fc.__repr__()


def test_not_tree_clone():
    net = pax.nn.Sequential(
        pax.nn.Linear(2, 3),
        jax.nn.relu,
        pax.nn.Linear(3, 4),
        jnp.tanh,
        pax.nn.Linear(4, 2),
        jax.nn.one_hot,
    )
    net = net.copy()


def test_class_attribute_copy():
    class M(pax.Module):
        a_list = [1, 2]

        def __init__(self):
            super().__init__()
            self.fc = pax.nn.Linear(3, 3)

    m = M()
    print(m.__class__.__dict__)
    m1 = m.copy()
    m.a_list.append(3)
    assert m.a_list == m1.a_list


def test_assign_empty_list_dict():
    fc = pax.nn.Linear(3, 3)

    fc.a = []

    fc.a.append(1)
    assert fc.a == [1]
    del fc.a[0]

    fc.b = {}
    fc.b[1] = 2


def test_automatic_assign_module_list_1():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = []
            for i in range(5):
                self.fc.append(pax.nn.Linear(3, 3))

    m = M()
    m = pax.scan_bugs(m)


def test_automatic_assign_module_dict_1():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = {}
            for i in range(5):
                self.fc[i] = pax.nn.Linear(3, 3)

    m = M()
    m = pax.scan_bugs(m)


def test_assign_empty_list_2():
    class M(pax.Module):
        fc: List[Module]

        def __init__(self):
            super().__init__()
            self.register_modules("fc", [])
            for i in range(5):
                self.fc.append(pax.nn.Linear(3, 3))

    m = M()
    m = pax.scan_bugs(m)


def test_compare_modules():
    a = pax.nn.Sequential(pax.nn.Linear(3, 3), pax.nn.Linear(4, 4))
    b = a.copy()
    assert a == b
    assert pax.enable_eval_mode(a) != b
    assert pax.freeze_parameters(a) != b
    assert pax.unfreeze_parameters(pax.freeze_parameters(a), origin=a) == b


def test_apply_inside_state_subtree():
    class M2(pax.Module):
        m2: Dict[str, Any]

        def __init__(self, m11):
            super().__init__()
            self.register_states("m2", {"m1": m11})

    m2 = M2(pax.nn.Linear(2, 2))
    assert m2.is_training() == True
    assert m2.m2["m1"].is_training() == True
    m2 = pax.enable_eval_mode(m2)
    assert m2.is_training() == False
    assert m2.m2["m1"].is_training() == True


def test_hash_module():
    a = pax.nn.LSTM(3, 3)
    b = a.copy()
    assert hash(a) == hash(b)


def test_deepcopy_pytreedef():
    f = pax.nn.Linear(3, 3)

    f.de = jax.tree_structure(f)

    g = f.copy()

    assert jax.tree_structure(g) == jax.tree_structure(f)


def test_delete_attribute():
    f = pax.nn.Linear(3, 3)

    f.t = pax.nn.Linear(1, 1)

    assert "t" in f._pax.name_to_kind
    with pytest.raises(ValueError):
        del f.t


def test_module_list_contains_int():
    class M(pax.Module):
        lst: List[Module]

        def __init__(self):
            super().__init__()

            self.register_modules("lst", [])
            self.lst.append(pax.nn.Linear(3, 3))
            self.lst.append(0)  # type: ignore

    with pytest.raises(ValueError):
        m = M()


def test_append_module_list():
    n = pax.nn.Sequential(pax.nn.Linear(3, 3))
    n.modules.append(pax.nn.Linear(4, 4))
    pax.scan_bugs(n)
