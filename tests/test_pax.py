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
        count: Union[int, pax.State]

        def __init__(self):
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == []


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
        count: Sequence[pax.State]

        def __init__(self):
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_type_dict():
    class Counter(pax.Module):
        count: Dict[str, pax.State]

        def __init__(self):
            self.count = {"conv1": [1, 2, 3], "conv2": ["a", "b"]}

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, pax.State]]

        def __init__(self):
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [1, 2, 3, "a", "b"]


def test_type_dict_dict_optional():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, Optional[pax.State]]]

        def __init__(self):
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

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
        count: Optional[pax.State]

        def __init__(self):
            self.count = 0

    counter = Counter()
    leaves, treedef = jax.tree_flatten(counter)
    assert leaves == [0]


def test_train_eval():
    net = pax.nn.Sequential(pax.nn.Linear(3, 3), pax.nn.Linear(3, 3))

    assert net._training == True
    net.pytree_layers[0]._training == True
    net = net.eval()
    assert net._training == False
    net.pytree_layers[0]._training == False
    net.pytree_layers[1]._training == False
    net = net.train()
    assert net._training == True
    net.pytree_layers[0]._training == True
    net.pytree_layers[1]._training == True
