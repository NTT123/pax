"""
Manage thread local states
"""

import random
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util

KeyArray = Union[Any, jnp.ndarray]


class PaxThreadingLocalState(threading.local):
    """Manage all thread local states used by PAX"""

    __slots__ = [
        "_mutable_module_ref_list",
        "_mutable_module_level",
        "_rng",
    ]
    _mutable_module_ref_list: Tuple[weakref.ReferenceType, ...]
    _mutable_module_level: jax.core.Sublevel
    _rng: Optional[random.Random]

    def __init__(self):
        super().__init__()
        self._mutable_module_ref_list = ()
        self._mutable_module_level = jax.core.cur_sublevel()
        self._rng = random.Random(42)

    def add_mutable_module(self, module):
        """add `module` to mutable list"""
        self._mutable_module_ref_list = (
            weakref.ref(module),
            *self._mutable_module_ref_list,
        )

    def is_mutable(self, module):
        """Is `module` mutable?"""

        # cannot modify a module whose level of abstraction
        # is lower than the current level
        if self._mutable_module_level < jax.core.cur_sublevel():
            return False

        for ref in self._mutable_module_ref_list:
            if module is ref():
                return True

        return False

    @contextmanager
    def allow_mutation(self, modules):
        """A context manager that turns on mutability."""

        if not isinstance(modules, (tuple, list)):
            modules = (modules,)
        modules = tuple(weakref.ref(mod) for mod in modules)

        prev = self._mutable_module_ref_list
        prev_abstraction_level = self._mutable_module_level
        try:
            self._mutable_module_ref_list = modules
            self._mutable_module_level = jax.core.cur_sublevel()
            yield
        finally:
            self._mutable_module_ref_list = prev
            self._mutable_module_level = prev_abstraction_level

    def seed_rng_key(self, seed: int) -> None:
        """Set ``self._rng = random.Random(seed)``.

        Arguments:
            seed: an integer seed.
        """
        assert isinstance(seed, int)
        self._rng = random.Random(seed)

    def next_rng_key(self) -> KeyArray:
        """Return a random rng key. Renew the global random state."""
        seed = self._rng.randint(1, 999999999)
        return jax.random.PRNGKey(seed)

    def get_rng_state(self):
        """Return internal random states."""
        return self._rng.getstate()

    def set_rng_state(self, state):
        """Set internal random states."""
        self._rng.setstate(state)


PAX_STATE = PaxThreadingLocalState()
add_mutable_module = PAX_STATE.add_mutable_module
allow_mutation = PAX_STATE.allow_mutation
get_rng_state = PAX_STATE.get_rng_state
is_mutable = PAX_STATE.is_mutable
next_rng_key = PAX_STATE.next_rng_key
seed_rng_key = PAX_STATE.seed_rng_key
set_rng_state = PAX_STATE.set_rng_state
