"""Test important pax stuffs."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pax


def test_pax_next_rng_key():
    # seed 42
    pax.seed_rng_key(42)
    assert pax.rng.state._rng_key is None
    assert pax.rng.state._seed == 42
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
            self.register_parameter_subtree("count", [0])

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_type_dict():
    class Counter(pax.Module):
        count: Dict[str, int]

        def __init__(self):
            super().__init__()
            self.register_state_subtree(
                "count", {"conv1": [1, 2, 3], "conv2": ["a", "b"]}
            )

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, int]]

        def __init__(self):
            super().__init__()
            self.register_state_subtree(
                "count", {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}
            )

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict_optional():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, Optional[int]]]

        def __init__(self):
            super().__init__()
            self.register_state_subtree(
                "count", {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}
            )

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict_optional1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, Optional[int]]]

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
            self.register_state("count", 0)

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_train_eval():
    net = pax.nn.Sequential(pax.nn.Linear(3, 3), pax.nn.Linear(3, 3))

    assert net._training == True
    net = net.eval()
    assert net._training == False
    assert net.modules[0]._training == False
    assert net.modules[1]._training == False
    net = net.train()
    assert net._training == True
    assert net.modules[0]._training == True
    assert net.modules[1]._training == True


def test_state_of_param():
    class M1(pax.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("p1", jnp.array(0.0, dtype=jnp.float32))

    m1 = M1()

    class M2(pax.Module):
        def __init__(self, m11):
            super().__init__()
            self.register_state_subtree("m2", {"m1": m11})

    m2 = M2(m1)
    assert len(jax.tree_leaves(m1.filter("state"))) == 0
    assert len(jax.tree_leaves(m2.filter("parameter"))) == 0


def test_module_properties_modify():
    fc = pax.nn.Linear(3, 3)
    assert fc._training == True
    fc1 = fc.copy()
    assert fc1._training == True
    fc = fc.eval()
    assert fc._training == False
    assert fc1._training == True


def test_clone_no_side_effect():
    fc1 = pax.nn.Linear(3, 3)
    fc2 = fc1.copy()
    fc1.new_module = pax.nn.Linear(5, 5)
    assert "new_module" in fc1._name_to_kind  # registered 'new_modules' as part of fc1
    assert (
        "new_module" not in fc2._name_to_kind
    )  # fc2._name_to_kind is different from fc1._name_to_kind
