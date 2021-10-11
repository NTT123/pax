"""
Manage the global variable ``_state._rng_key``. Generate new ``rng_key`` if requested.
"""
import logging
import threading
from typing import Any, Union

import jax
import jax.numpy as jnp
import jax.tree_util

_state = threading.local()
_state._rng_key = None
_state._seed = None


KeyArray = Union[Any, jnp.ndarray]


def seed_rng_key(seed: int) -> None:
    """Set ``state._seed = seed`` and reset ``state._rng_key`` to ``None``.

    Arguments:
        seed: an interger seed.
    """
    assert isinstance(seed, int)
    _state._seed = seed
    _state._rng_key = None  # reset `_rng_key`


def next_rng_key() -> KeyArray:
    """Return a random rng key. Renew the global random key ``state._rng_key``.

    If ``state._rng_key`` is ``None``, generate a new ``state._rng_key`` from ``state._seed``.
    """
    if _state._rng_key is None:
        if _state._seed is None:
            seed = 42
            logging.warning(
                f"Seeding RNG key with seed {seed}. "
                f"Use `pax.seed_rng_key` function to avoid this warning."
            )
            seed_rng_key(seed)

        # Delay the generating of state._rng_key until `next_rng_key` is called.
        # This helps to avoid the problem when `seed_rng_key` is called
        # before jax found TPU cores.
        if _state._seed is not None:
            _state._rng_key = jax.random.PRNGKey(_state._seed)
        else:
            raise ValueError("Impossible")

    key, _state._rng_key = jax.random.split(_state._rng_key)

    return key


__all__ = ["seed_rng_key", "next_rng_key"]
