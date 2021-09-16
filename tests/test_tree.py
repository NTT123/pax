import jax
import jax.numpy as jnp
import pax


def test_node_1():
    n = pax.tree.Node()
    print(jax.tree_flatten(n))


def test_pax_list():
    n = pax.tree.PaxList([1, 2, 3])
    print(jax.tree_flatten(n))
    n.append(4)
    print(jax.tree_flatten(n))


def test_pax_list_of_list():
    n = pax.tree.to_tree([1, 2, [3, 4, ["hello"]]])
    print(n)
    print(jax.tree_flatten(n))


def test_pax_dict():
    n = pax.tree.PaxDict({1: 2, 2: 3})
    print(n)
    print(jax.tree_flatten(n))


def test_pax_dict_of_list():
    n = pax.tree.Tree({1: 2, 2: [3, 4, [5, 6]]})
    print(n)
    a, b = jax.tree_flatten(n)
    c = jax.tree_unflatten(b, a)
    print(c)


def test_tree_params():
    class M(pax.Module):
        def __init__(self):
            super().__init__()

            self.f = pax.tree.ModuleTree(pax.nn.Linear(3, 3))

        def __call__(self, x):
            return x * self.f

    m = M()
