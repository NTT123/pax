"""Test important pax stuffs."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jmp
import pax
import pytest
from pax import Module


def test_pax_next_rng_key():
    # seed 42
    pax.seed_rng_key(42)
    # assert pax.rng.RNG_STATE.rng_key is None
    # assert pax.rng.RNG_STATE.rng_seed == 42
    expected_rng = jnp.array([0, 42], dtype=jnp.uint32)
    rng1 = pax.next_rng_key()
    expected_rng_1, rng_internal = jax.random.split(expected_rng)
    # np.testing.assert_array_equal(rng1, expected_rng_1)
    rng2 = pax.next_rng_key()
    expected_rng_2, rng_internal = jax.random.split(rng_internal)
    # np.testing.assert_array_equal(rng2, expected_rng_2)


def test_type_union():
    class Counter(pax.Module):
        count: Union[int, jnp.ndarray]

        def __init__(self):
            super().__init__()
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_list_int():
    class Counter(pax.Module):
        count: List[int]

        def __init__(self):
            super().__init__()
            self.count = [0]

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_sequence():
    class Counter(pax.StateModule):
        count: Sequence[int]

        def __init__(self):
            super().__init__()
            self.count = jnp.array([0.0])

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == [jnp.array(0.0)]


def test_type_dict():
    class Counter(pax.Module):
        def __init__(self):
            super().__init__()
            self.count = {"conv1": [1, 2, 3], "conv2": ["a", "b"]}

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_dict_dict1():
    class Counter(pax.Module):
        def __init__(self):
            super().__init__()
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_dict_dict_optional():
    class Counter(pax.Module):
        def __init__(self):
            super().__init__()
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_dict_dict_optional1():
    class Counter(pax.Module):
        count: Dict[str, Dict[int, List[Union[int, str]]]]

        def __init__(self):
            super().__init__()
            self.count = {"conv1": {1: [1, 2, 3]}, "conv2": {2: ["a", "b"]}}

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_tuple():
    class Counter(pax.Module):
        count: Tuple[int, int]

        def __init__(self):
            super().__init__()
            self.count = (1, 2)

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == []


def test_type_optional():
    class Counter(pax.Module):
        count: Optional[int]

        def __init__(self):
            super().__init__()
            self.count = jnp.array(0)

    counter = Counter()
    leaves, treedef = jax.tree_util.tree_flatten(counter)
    assert leaves == [0]


def test_train_eval():
    net = pax.Sequential(pax.Linear(3, 3), pax.Linear(3, 3))

    assert net.training == True
    net = pax.enable_eval_mode(net)
    assert net.training == False
    assert net.modules[0].training == False
    assert net.modules[1].training == False
    net = pax.enable_train_mode(net)
    assert net.training == True
    assert net.modules[0].training == True
    assert net.modules[1].training == True


def test_state_of_param():
    class M1(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.p1 = jnp.array(0.0, dtype=jnp.float32)

    m1 = M1()

    class M2(pax.StateModule):
        def __init__(self, m11):
            super().__init__()
            self.m2 = {"m1": m11}

    m2 = M2(m1)
    assert len(jax.tree_util.tree_leaves(pax.select_parameters(m2))) == 1
    assert len(jax.tree_util.tree_leaves(pax.select_parameters(m1))) == 1


def test_assign_module_with_default_kind():
    class M1(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    _ = M1()

    class M2(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    _ = M2()

    class M11(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    _ = M11()

    class M22(pax.StateModule):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    _ = M22()


def test_default_kind_module():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    m = M()
    # assert m.pax.name_to_kind["fc"] is pax.PaxKind.MODULE


def test_default_kind_attribute_order():
    class M(pax.Module):
        parameters = pax.parameters_method("b", "t", "c")

        def __init__(self):
            super().__init__()
            self.e = jnp.array(0)
            self.a = jnp.array(0)
            self.z = jnp.array(0)

            self.b = jnp.array(0.0)
            self.t = jnp.array(0.0)
            self.c = jnp.array(0.0)

    for _ in range(1000):
        m = M()
        # assert tuple(m.pax.name_to_kind.keys()) == ("e", "a", "z", "b", "t", "c")


def test_module_properties_modify():
    fc = pax.Linear(3, 3)
    assert fc.training == True
    fc1 = fc.copy()
    assert fc1.training == True
    fc = pax.enable_eval_mode(fc)
    assert fc.training == False
    assert fc1.training == True


def test_clone_no_side_effect():
    fc1 = pax.BatchNorm1D(3)
    fc2 = fc1.copy()
    fc1 = fc1.set_attribute("new_module", pax.Linear(5, 5))

    # assert (
    #     "new_module" in fc1.pax.name_to_kind
    # ), "registered 'new_modules' as part of fc1"
    # assert (
    #     "new_module" not in fc2.pax.name_to_kind
    # ), "fc2.pax.name_to_kind is different from fc1.pax.name_to_kind"


def test_lambda_module():
    f = pax.Lambda(jax.nn.relu)
    x = jnp.array(5.0)
    y = f(x)
    assert x.item() == y.item()

    x = jnp.array(-4.0)
    y = f(x)
    assert y.item() == 0.0


def test_forget_call_super_at_init():
    class M(pax.Module):
        def __init__(self):
            self.fc = pax.Linear(3, 3)

    # with initialization in the `__new__` method,
    # no need to call `super().__init__()` anymore.
    m = M()


def test_name_repr():
    fc = pax.Linear(2, 3, name="fc1")
    assert "(fc1)" in fc.__repr__()


def test_not_tree_clone():
    net = pax.Sequential(
        pax.Linear(2, 3),
        jax.nn.relu,
        pax.Linear(3, 4),
        jnp.tanh,
        pax.Linear(4, 2),
        jax.nn.one_hot,
    )
    net = net.copy()


def test_class_attribute_copy():
    class M(pax.Module):
        a_list = [1, 2]

        def __init__(self):
            super().__init__()
            self.fc = pax.Linear(3, 3)

    m = M()
    print(m.__class__.__dict__)
    m1 = m.copy()
    m.a_list.append(3)
    assert m.a_list == m1.a_list


def test_assign_empty_list_dict():
    fc = pax.Linear(3, 3)
    fc = fc.set_attribute("a", [])
    fc.a.append(1)  # type: ignore
    assert fc.a == [1]  # type: ignore
    del fc.a[0]  # type: ignore

    fc = fc.set_attribute("b", {})
    fc.b[1] = 2  # type: ignore


def test_automatic_assign_module_list_1():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = []
            for i in range(5):
                self.fc.append(pax.Linear(3, 3))

    m = M()
    m.scan_bugs()


def test_automatic_assign_module_dict_1():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.fc = {}
            for i in range(5):
                self.fc[i] = pax.Linear(3, 3)

    m = M()
    m.scan_bugs()


def test_assign_empty_list_2():
    class M(pax.Module):
        fc: List[Module]

        def __init__(self):
            super().__init__()
            self.fc = []
            for i in range(5):
                self.fc.append(pax.Linear(3, 3))

    m = M()
    m.scan_bugs()


def test_compare_modules():
    a = pax.Sequential(pax.Linear(3, 3), pax.Linear(4, 4))
    b = a.copy()
    assert a == b
    assert pax.enable_eval_mode(a) != b
    assert pax.freeze_parameters(a) != b
    # assert pax.unfreeze_parameters(pax.freeze_parameters(a), origin=a) == b


def test_apply_inside_state_subtree():
    class M2(pax.Module):
        m2: Dict[str, Any]

        def __init__(self, m11):
            super().__init__()
            self.m2 = {"m1": m11}

    m2 = M2(pax.Linear(2, 2))
    assert m2.training == True
    assert m2.m2["m1"].training == True
    m2 = pax.enable_eval_mode(m2)
    assert m2.training == False
    assert m2.m2["m1"].training == False


def test_hash_module():
    a = pax.LSTM(3, 3)
    b = a.copy()
    assert hash(a) == hash(b)


def test_deepcopy_pytreedef():
    f = pax.Linear(3, 3)
    f = f.set_attribute("de", jax.tree_util.tree_structure(f))
    g = f.copy()

    assert jax.tree_util.tree_structure(g) == jax.tree_util.tree_structure(f)


@pax.pure
def test_delete_attribute_1():
    f = pax.BatchNorm1D(3)
    f = f.set_attribute("t", pax.Linear(1, 1))
    # assert "t" in f.pax.name_to_kind
    with pytest.raises(ValueError):
        del f.t


def test_delete_attribute_2():
    def mutate(f: pax.Linear):
        del f.in_dim
        return f

    f = pax.Linear(3, 3)
    pax.pure(mutate)(f)


def test_module_list_contains_int():
    class M(pax.Module):
        lst: List[Module]

        def __init__(self):
            super().__init__()

            self.lst = []
            self.lst.append(pax.Linear(3, 3))
            self.lst.append(0)  # type: ignore

    # with pytest.raises(ValueError):
    m = M()


def test_append_module_list():
    n = pax.Sequential(pax.Linear(3, 3))
    n.replace(modules=n.modules + (pax.Linear(4, 4),))
    n.scan_bugs()


def test_replace_leaf():
    a = pax.Sequential(pax.Linear(2, 2), pax.Linear(2, 3))
    a = a.replace_node(a[0].weight, jnp.zeros((3, 2)))
    assert a[0].weight.shape == (3, 2)


def test_replace_node():
    a = pax.Sequential(pax.Linear(2, 2), pax.Linear(2, 3))
    relu = pax.Lambda(jax.nn.relu)
    a = a.replace_node(a[1], relu)
    assert a[1] is relu
    print(a.summary())


def test_replace_no_node():
    a = pax.Sequential(pax.Linear(2, 2), pax.Linear(2, 3))
    relu = pax.Lambda(jax.nn.relu)
    with pytest.raises(ValueError):
        a = a.replace_node(3, relu)


def test_shared_weight():
    fc1 = pax.Linear(2, 3)
    fc2 = pax.Linear(2, 3)
    fc2 = fc2.replace(weight=fc1.weight)
    with pytest.raises(ValueError):
        _ = pax.Sequential(fc1, fc2)


def test_shared_module():
    class M(pax.Module):
        def __init__(self):
            super().__init__()

            mod = pax.Linear(3, 3)
            self.fc1 = mod
            self.fc2 = mod

    with pytest.raises(ValueError):
        _ = M()


def test_class_with_property():
    class M(pax.StateModule):
        @property
        def counter(self):
            return self._counter

        def __init__(self):
            super().__init__()
            self._counter = jnp.array(0)

        def __call__(self):
            self._counter += 1

    m = M()
    m, _ = pax.module_and_value(m)()
    print(m.counter)


def test_mix_pytree_and_nonpytree():
    class Mix(pax.Module):
        def __init__(self):
            super().__init__()
            self.funcs = []
            self.funcs.append(pax.Linear(3, 3))
            self.funcs.append(jax.nn.relu)
            self.weight = jnp.array(0.0)
            self.count = jnp.array(0)

        parameters = pax.parameters_method("weight")

    net = Mix()
    net = jmp.cast_to_half(net)
    params = net.parameters()
    net.update_parameters(params)


def test_parameter_module():
    class C(pax.StateModule):
        def __init__(self):
            super().__init__()
            self.b = jnp.array(0)

    class D(pax.ParameterModule):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = C()

    d = D()
    assert len(jax.tree_util.tree_leaves(d.parameters())) == 0


def test_state_dict_mixed_attribute():
    class Counter(pax.ParameterModule):
        def __init__(self):
            super().__init__()
            self.f = [jnp.array(0), jax.nn.relu]

    c = Counter()
    st = c.state_dict()
    assert st["f"][-1] is None
    c = c.load_state_dict(st)
    assert c.f[-1] is jax.nn.relu


def test_uninitialized_node():
    class Att(pax.Module):
        def __init__(self):
            self.hidden = pax.EmptyNode()

    att = Att()
    assert att.parameters() == att
    assert att.pytree_attributes == ("hidden",)
    assert att.hidden == pax.EmptyNode()


def test_dataclass_eval():
    @dataclass
    class M(pax.Module):
        core: pax.Module

    m = M(pax.Linear(3, 3))
    m = m.eval()


def test_mutate_inside_scan():
    class M(pax.Module):
        def __init__(self):
            super().__init__()
            self.rng = pax.RngSeq()

        def __call__(self, x):
            def loop(prev, inp):
                rng_key = self.rng.next_rng_key()
                o = inp + jax.random.normal(rng_key, inp.shape)
                return prev, o

            state = jnp.zeros((x.shape[0], 1))
            state, y = pax.scan(loop, state, x, time_major=False)
            return y

    net = M()
    x = jnp.ones((2, 50, 1))
    f = jax.jit(pax.pure(lambda net, x: net(x)))
    with pytest.raises(ValueError):
        f(net, x)
