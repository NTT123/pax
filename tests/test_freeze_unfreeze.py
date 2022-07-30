import jax
import pax


def test_freeze_really_working():
    a = pax.Sequential(
        pax.Linear(3, 3),
        pax.Linear(5, 5),
    )
    b = pax.freeze_parameters(a)
    # assert b[0].pax.name_to_kind["weight"] == pax.PaxKind.STATE
    # assert a[0].pax.name_to_kind["weight"] == pax.PaxKind.PARAMETER


def test_freeze_mapping_proxy():
    a = pax.Sequential(
        pax.Linear(3, 3),
        pax.Linear(5, 5),
    )
    b = pax.freeze_parameters(a)
    # assert isinstance(b.pax.name_to_kind, MappingProxyType), "expecting a proxy map"


def test_freeze_twice():
    a = pax.Linear(2, 2)
    # with pytest.raises(ValueError):
    _ = pax.freeze_parameters(pax.freeze_parameters(a))


# def test_freeze_unfreeze():
#     a = pax.Sequential(
#         pax.Linear(2, 2),
#         pax.Linear(3, 3),
#         pax.Linear(4, 4),
#         pax.Linear(5, 5),
#     )

#     b = pax.freeze_parameters(a)
#     c = pax.unfreeze_parameters(b, origin=a)
#     # pylint: disable=-access
#     # assert a[0].pax.name_to_kind is c[0].pax.name_to_kind


def test_copy():
    a = pax.Linear(1, 1, with_bias=False)
    b = pax.enable_eval_mode(a)
    assert jax.tree_util.tree_structure(a) != jax.tree_util.tree_structure(b)
    c = pax.enable_train_mode(b)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(c)
