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
RNG_STATE.rng_key = None
RNG_STATE.rng_seed = None


KeyArray = Union[Any, jnp.ndarray]


def get_rng_state():
    """Return internal states."""
    return (RNG_STATE.rng_key, RNG_STATE.rng_seed)


def set_rng_state(state):
    """Set internal states."""
    rng_key, seed = state
    RNG_STATE.rng_key = rng_key
    RNG_STATE.rng_seed = seed


def seed_rng_key(seed: int) -> None:
    """Set ``RNG_STATE.rng_seed = seed`` and reset ``RNG_STATE.rng_key`` to ``None``.

    Arguments:
        seed: an integer seed.
    """
    assert isinstance(seed, int)
    RNG_STATE.rng_seed = seed
    RNG_STATE.rng_key = None  # reset `rng_key`


def next_rng_key() -> KeyArray:
    """Return a random rng key. Renew the global random key ``RNG_STATE.rng_key``.

    If ``RNG_STATE.rng_key`` is ``None``,
    generate a new ``RNG_STATE.rng_key`` from ``RNG_STATE.rng_seed``.
    """
    if RNG_STATE.rng_key is None:
        if RNG_STATE.rng_seed is None:
            seed = 42
            logging.warning(
                "Seeding RNG key with seed %s. "
                "Use `pax.seed_rng_key` function to avoid this warning.",
                seed,
            )
            seed_rng_key(seed)

        # Delay the generating of RNG_STATE.rng_key until `next_rng_key` is called.
        # This helps to avoid the problem when `seed_rng_key` is called
        # before jax found TPU cores.
        if RNG_STATE.rng_seed is not None:
            RNG_STATE.rng_key = jax.random.PRNGKey(RNG_STATE.rng_seed)
        else:
            raise ValueError("Impossible")

    key, RNG_STATE.rng_key = jax.random.split(RNG_STATE.rng_key)

    return key


__all__ = ["seed_rng_key", "next_rng_key"]
