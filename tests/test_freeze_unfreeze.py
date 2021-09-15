from types import MappingProxyType

import jax
import pax
import pytest


def test_freeze_really_working():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = a.freeze()
    assert b[0]._name_to_kind["weight"] == pax.module.PaxFieldKind.STATE
    assert a[0]._name_to_kind["weight"] == pax.module.PaxFieldKind.PARAMETER
    assert b[0]._name_to_kind_to_unfreeze["weight"] == pax.module.PaxFieldKind.PARAMETER
    assert a[0]._name_to_kind_to_unfreeze is None


def test_freeze_mapping_proxy():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = a.freeze()
    assert isinstance(b._name_to_kind, MappingProxyType), "expecting a proxy map"


def test_freeze_twice():
    a = pax.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        b = a.freeze().freeze()


def test_freeze_unfreeze():
    a = pax.nn.Sequential(
        pax.nn.Linear(2, 2),
        pax.nn.Linear(3, 3),
        pax.nn.Linear(4, 4),
        pax.nn.Linear(5, 5),
    )
    b = a.freeze()
    c = b.unfreeze()
    assert a[0]._name_to_kind is c[0]._name_to_kind


def test_copy():
    a = pax.nn.Linear(1, 1, with_bias=False)
    b = a.eval().train()
    assert jax.tree_structure(a) == jax.tree_structure(b)
