import time

import jax
import numpy as np
import pax


def test_perf_transformer_flatten_unflatten():
    class MyTransformer(pax.Module):
        def __init__(self, num_layers: int):
            super().__init__()
            self.layers = [
                pax.nn.MultiHeadAttention(8, 512 // 8, 1.0) for i in range(num_layers)
            ]

    f = MyTransformer(16)

    start = time.perf_counter()
    n_iters = 10_000
    for i in range(n_iters):
        leaves, treedef = jax.tree_flatten(f)
        f = jax.tree_unflatten(treedef, leaves)
    end = time.perf_counter()
    iters_per_second = n_iters / (end - start)
    print(iters_per_second, "iters/second")
    assert iters_per_second > 2500


def test_perf_resnet200_flatten_unflatten():

    f = pax.nets.ResNet200(3, 100)

    start = time.perf_counter()
    n_iters = 1000
    for i in range(n_iters):
        leaves, treedef = jax.tree_flatten(f)
        f = jax.tree_unflatten(treedef, leaves)
    end = time.perf_counter()
    iters_per_second = n_iters / (end - start)
    print(iters_per_second, "iters/second")
    assert iters_per_second > 100


def test_perf_flattenmodule_resnet200_flatten_unflatten():

    x = jax.random.normal(jax.random.PRNGKey(42), (1, 3, 64, 64))
    f = pax.nets.ResNet200(3, 100)
    y = f(x)
    f = pax.flatten_module(f)
    y1 = f(x)

    np.testing.assert_array_equal(y, y1)

    start = time.perf_counter()
    n_iters = 1000
    for i in range(n_iters):
        leaves, treedef = jax.tree_flatten(f)
        f = jax.tree_unflatten(treedef, leaves)
    end = time.perf_counter()
    iters_per_second = n_iters / (end - start)
    print(iters_per_second, "iters/second")
    assert iters_per_second > 100
