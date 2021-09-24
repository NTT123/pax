from types import MappingProxyType

import jax
import pax
import pytest


def test_freeze_really_working():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = pax.freeze_parameter(a)
    assert b[0]._name_to_kind["weight"] == pax.module.PaxFieldKind.STATE
    assert a[0]._name_to_kind["weight"] == pax.module.PaxFieldKind.PARAMETER


def test_freeze_mapping_proxy():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = pax.freeze_parameter(a)
    assert isinstance(b._name_to_kind, MappingProxyType), "expecting a proxy map"


def test_freeze_twice():
    a = pax.nn.Linear(2, 2)
    # with pytest.raises(ValueError):
    b = pax.freeze_parameter(pax.freeze_parameter(a))


def test_freeze_unfreeze():
    a = pax.nn.Sequential(
        pax.nn.Linear(2, 2),
        pax.nn.Linear(3, 3),
        pax.nn.Linear(4, 4),
        pax.nn.Linear(5, 5),
    )

    b = pax.freeze_parameter(a)
    c = pax.unfreeze_parameter(b, origin=a)
    assert a[0]._name_to_kind is c[0]._name_to_kind


def test_copy():
    a = pax.nn.Linear(1, 1, with_bias=False)
    b = pax.enable_eval_mode(a)
    assert jax.tree_structure(a) != jax.tree_structure(b)
    c = pax.enable_train_mode(b)
    assert jax.tree_structure(a) == jax.tree_structure(c)
