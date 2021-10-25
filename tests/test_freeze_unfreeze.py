from types import MappingProxyType

import jax
import pax


def test_freeze_really_working():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = pax.freeze_parameters(a)
    # pylint: disable=protected-access
    assert b[0]._pax.name_to_kind["weight"] == pax.PaxKind.STATE
    # pylint: disable=protected-access
    assert a[0]._pax.name_to_kind["weight"] == pax.PaxKind.PARAMETER


def test_freeze_mapping_proxy():
    a = pax.nn.Sequential(
        pax.nn.Linear(3, 3),
        pax.nn.Linear(5, 5),
    )
    b = pax.freeze_parameters(a)
    # pylint: disable=protected-access
    assert isinstance(b._pax.name_to_kind, MappingProxyType), "expecting a proxy map"


def test_freeze_twice():
    a = pax.nn.Linear(2, 2)
    # with pytest.raises(ValueError):
    _ = pax.freeze_parameters(pax.freeze_parameters(a))


def test_freeze_unfreeze():
    a = pax.nn.Sequential(
        pax.nn.Linear(2, 2),
        pax.nn.Linear(3, 3),
        pax.nn.Linear(4, 4),
        pax.nn.Linear(5, 5),
    )

    b = pax.freeze_parameters(a)
    c = pax.unfreeze_parameters(b, origin=a)
    # pylint: disable=protected-access
    assert a[0]._pax.name_to_kind is c[0]._pax.name_to_kind


def test_copy():
    a = pax.nn.Linear(1, 1, with_bias=False)
    b = pax.enable_eval_mode(a)
    assert jax.tree_structure(a) != jax.tree_structure(b)
    c = pax.enable_train_mode(b)
    assert jax.tree_structure(a) == jax.tree_structure(c)
