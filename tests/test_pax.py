"""Test important pax stuffs."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pax


def test_pax_next_rng_key():
    # seed 42
    pax.seed_rng_key(42)
    expected_rng = jnp.array([0, 42], dtype=jnp.uint32)
    np.testing.assert_array_equal(pax.rng.state._rng_key, expected_rng)
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
            self.register_state("count", [0])

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_type_list_int():
    class Counter(pax.Module):
        count: List[int]

        def __init__(self):
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_sequence():
    class Counter(pax.Module):
        count: Sequence[int]

        def __init__(self):
            self.register_param_subtree("count", [0])

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_type_dict():
    class Counter(pax.Module):
        count: Dict[str, int]

        def __init__(self):
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
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_tuple():
    class Counter(pax.Module):
        count: Tuple[int, int]

        def __init__(self):
            self.count = (1, 2)

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


def test_type_optional():
    class Counter(pax.Module):
        count: Optional[int]

        def __init__(self):
            self.register_state("count", 0)

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_train_eval():
    net = pax.nn.Sequential(pax.nn.Linear(3, 3), pax.nn.Linear(3, 3))

    assert net._training == True
    net.modules[0]._training == True
    net = net.eval()
    assert net._training == False
    net.modules[0]._training == False
    net.modules[1]._training == False
    net = net.train()
    assert net._training == True
    net.modules[0]._training == True
    net.modules[1]._training == True


def test_state_of_param():
    class M1(pax.Module):
        def __init__(self):
            self.register_parameter("p1", jnp.array(0.0, dtype=jnp.float32))

    m1 = M1()

    class M2(pax.Module):
        def __init__(self, m11):
            self.register_state_subtree("m2", {"m1": m11})

    m2 = M2(m1)
    assert len(jax.tree_leaves(m1.filter("state"))) == 0
    assert len(jax.tree_leaves(m2.filter("parameter"))) == 0
