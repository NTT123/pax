"""Useful functions."""

from typing import TypeVar
from unittest import TestCase

import jax

from .module import Module

T = TypeVar("T", bound=Module)


def assert_structure_equal(tree_a: T, tree_b: T):
    """Assert that the two pytrees are structurally the same.

    Print out the difference.
    """
    if jax.tree_structure(tree_a) == jax.tree_structure(tree_b):
        return True

    def check(subtree_a, subtree_b):
        if isinstance(subtree_a, Module) and isinstance(subtree_b, Module):
            assert_structure_equal(subtree_a, subtree_b)

    has_error = False
    try:
        jax.tree_map(
            check,
            tree_a,
            tree_b,
            is_leaf=lambda x: isinstance(x, Module)
            and x is not tree_a
            and x is not tree_b,
        )
    except ValueError:
        has_error = True

    if has_error:
        test_case = TestCase()
        test_case.maxDiff = None
        # do not compare weights
        tree_a_w_none_leaves = jax.tree_map(lambda _: None, tree_a)
        tree_b_w_none_leaves = jax.tree_map(lambda _: None, tree_b)
        test_case.assertDictEqual(
            vars(tree_a_w_none_leaves), vars(tree_b_w_none_leaves)
        )

    return has_error
