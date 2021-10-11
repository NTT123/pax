from typing import TypeVar
from unittest import TestCase

import jax

from .module import Module

T = TypeVar("T", bound=Module)


def get_modules(v):
    "Return a list of modules in the pytree `v`."
    modules = jax.tree_flatten(v, is_leaf=lambda x: isinstance(x, Module))[0]
    modules = [m for m in modules if isinstance(m, Module)]
    return modules


def assertStructureEqual(self: T, other: T):
    """Assert that the two modules are structurally the same.

    Print out the difference.
    """
    if jax.tree_structure(self) == jax.tree_structure(other):
        return True

    def check(a, b):
        if isinstance(a, Module) and isinstance(b, Module):
            assertStructureEqual(a, b)

    has_error = False
    try:
        jax.tree_map(
            check,
            self,
            other,
            is_leaf=lambda x: isinstance(x, Module)
            and x is not self
            and x is not other,
        )
    except ValueError:
        has_error = True

    if has_error:
        tc = TestCase()
        tc.maxDiff = None
        # do not compare weights
        u = jax.tree_map(lambda x: None, self)
        v = jax.tree_map(lambda y: None, other)
        tc.assertDictEqual(vars(u), vars(v))
