"""
Manage thread local states
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util

KeyArray = Union[Any, jnp.ndarray]


class PaxThreadingLocalState(threading.local):
    """Manage all thread local states used by PAX"""

    __slots__ = [
        "_enable_deep_copy",
        "_inside_pure_function",
        "_mutable_module_id_list",
        "_rng_seed",
        "_rng_key",
    ]
    _enable_deep_copy: bool
    _inside_pure_function: bool
    _mutable_module_id_list: Tuple[int, ...]
    _rng_seed: Optional[int]
    _rng_key: Optional[KeyArray]

    def __init__(self):
        super().__init__()

        self._enable_deep_copy = False
        self._inside_pure_function = False
        self._mutable_module_id_list = ()
        self._rng_seed = None
        self._rng_key = None

    def is_inside_pure_function(self):
        """are we inside a function decorated by `pax.pure`"""
        return self._inside_pure_function

    def is_deep_copy_enabled(self):
        """use deepcopy to copy modules"""
        return self._enable_deep_copy

    def add_mutable_module(self, module):
        """add `module` to mutable list"""
        self._mutable_module_id_list = (id(module),) + self._mutable_module_id_list

    def is_mutable(self, module):
        """Is `module` mutable?"""
        return id(module) in self._mutable_module_id_list

    @contextmanager
    def enable_deep_copy(self):
        r"""A context manager that turns on deepcopy mode."""
        prev = self._enable_deep_copy
        self._enable_deep_copy = True
        try:
            yield
        finally:
            self._enable_deep_copy = prev

    @contextmanager
    def allow_mutation(self, modules):
        r"""A context manager that turns on mutability."""
        if not isinstance(modules, (tuple, list)):
            modules = (modules,)
        modules = tuple(id(mod) for mod in modules)

        prev = self._mutable_module_id_list
        prev_inside = self._inside_pure_function
        prev_rng_state = self.get_rng_state()
        try:
            self._inside_pure_function = True
            self._mutable_module_id_list = modules
            yield
        finally:
            self._mutable_module_id_list = prev
            self._inside_pure_function = prev_inside
            self.set_rng_state(prev_rng_state)

    def seed_rng_key(self, seed: int) -> None:
        """Set ``self.rng_seed = seed``.
        Reset ``self.rng_key`` to ``None``.

        Arguments:
            seed: an integer seed.
        """
        assert isinstance(seed, int)
        self._rng_seed = seed
        self._rng_key = None

    def next_rng_key(self) -> KeyArray:
        """Return a random rng key.
        Renew the global random key ``self.rng_key``.

        If ``self.rng_key`` is ``None``,
        generate a new ``self.rng_key`` from ``self.rng_seed``.
        """
        if self._rng_key is None:
            if self._rng_seed is None:
                seed = 42
                logging.warning(
                    "Seeding RNG key with seed %s. "
                    "Use `pax.seed_rng_key` function to avoid this warning.",
                    seed,
                )
                self.seed_rng_key(seed)

            # Delay the generating of self.rng_key until `next_rng_key` is called.
            # This helps to avoid the problem when `seed_rng_key` is called
            # before jax found TPU cores.
            if self._rng_seed is not None:
                self._rng_key = jax.random.PRNGKey(self._rng_seed)
            else:
                raise ValueError("Impossible")

        key, self._rng_key = jax.random.split(self._rng_key)

        return key

    def get_rng_state(self):
        """Return internal states."""
        return (self._rng_key, self._rng_seed)

    def set_rng_state(self, state):
        """Set internal states."""
        rng_key, seed = state
        self._rng_key = rng_key
        self._rng_seed = seed


PAX_STATE = PaxThreadingLocalState()
add_mutable_module = PAX_STATE.add_mutable_module
allow_mutation = PAX_STATE.allow_mutation
enable_deep_copy = PAX_STATE.enable_deep_copy
get_rng_state = PAX_STATE.get_rng_state
is_deep_copy_enabled = PAX_STATE.is_deep_copy_enabled
is_inside_pure_function = PAX_STATE.is_inside_pure_function
is_mutable = PAX_STATE.is_mutable
next_rng_key = PAX_STATE.next_rng_key
seed_rng_key = PAX_STATE.seed_rng_key
set_rng_state = PAX_STATE.set_rng_state
