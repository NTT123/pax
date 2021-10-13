"""
Manage the global variable ``RNG_STATE``. Generate new ``rng_key`` if requested.
"""

import logging
import threading
from typing import Any, Union

import jax
import jax.numpy as jnp
import jax.tree_util

RNG_STATE = threading.local()
RNG_STATE._rng_key = None
RNG_STATE._seed = None


KeyArray = Union[Any, jnp.ndarray]


def get_rng_state():
    """Return internal states."""
    return (RNG_STATE._rng_key, RNG_STATE._seed)


def set_rng_state(state):
    """Set internal states."""
    _rng_key, _seed = state
    RNG_STATE._rng_key = _rng_key
    RNG_STATE._seed = _seed


def seed_rng_key(seed: int) -> None:
    """Set ``state._seed = seed`` and reset ``state._rng_key`` to ``None``.

    Arguments:
        seed: an integer seed.
    """
    assert isinstance(seed, int)
    RNG_STATE._seed = seed
    RNG_STATE._rng_key = None  # reset `_rng_key`


def next_rng_key() -> KeyArray:
    """Return a random rng key. Renew the global random key ``state._rng_key``.

    If ``state._rng_key`` is ``None``, generate a new ``state._rng_key`` from ``state._seed``.
    """
    if RNG_STATE._rng_key is None:
        if RNG_STATE._seed is None:
            seed = 42
            logging.warning(
                f"Seeding RNG key with seed {seed}. "
                f"Use `pax.seed_rng_key` function to avoid this warning."
            )
            seed_rng_key(seed)

        # Delay the generating of state._rng_key until `next_rng_key` is called.
        # This helps to avoid the problem when `seed_rng_key` is called
        # before jax found TPU cores.
        if RNG_STATE._seed is not None:
            RNG_STATE._rng_key = jax.random.PRNGKey(RNG_STATE._seed)
        else:
            raise ValueError("Impossible")

    key, RNG_STATE._rng_key = jax.random.split(RNG_STATE._rng_key)

    return key


__all__ = ["seed_rng_key", "next_rng_key"]
