"""
Manage the global variable ``state._rng_key``. Generate new ``rng_key`` if requested.
"""
import logging
from typing import Any, Union

import jax
import jax.numpy as jnp
import jax.tree_util

from .ctx import state

KeyArray = Union[Any, jnp.ndarray]


def seed_rng_key(seed: int) -> None:
    """Set ``state._seed = seed`` and reset ``state._rng_key`` to ``None``.

    Arguments:
        seed: an interger seed.
    """
    assert isinstance(seed, int)
    state._seed = seed
    state._rng_key = None  # reset `_rng_key`


def next_rng_key() -> KeyArray:
    """Return a random rng key. Renew the global random key ``state._rng_key``.

    If ``state._rng_key`` is ``None``, generate a new ``state._rng_key`` from ``state._seed``.
    """
    if state._rng_key is None:
        if state._seed is None:
            seed = 42
            logging.warning(
                f"Seeding RNG key with seed {seed}. "
                f"Use `pax.seed_rng_key` function to avoid this warning."
            )
            seed_rng_key(seed)

        # Delay the generating of state._rng_key until `next_rng_key` is called.
        # This helps to avoid the problem when `seed_rng_key` is called
        # before jax found TPU cores.
        if state._seed is not None:
            state._rng_key = jax.random.PRNGKey(state._seed)
        else:
            raise ValueError("Impossible")

    key, state._rng_key = jax.random.split(state._rng_key)

    return key
