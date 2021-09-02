"""
Manage a global `state.rng_key`. Generate new rng_key if needed.
"""
import logging
import threading

import jax
import jax.numpy as jnp
import jax.tree_util

state = threading.local()
state._rng_key = None
state._seed = None


def seed_rng_key(seed: int) -> None:
    """Set ``state._seed = seed`` and reset ``_rng_key``."""
    state._seed = seed
    state._rng_key = None  # reset `_rng_key`


def next_rng_key() -> jnp.ndarray:
    """Return a random RNG key."""
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
        state._rng_key = jax.random.PRNGKey(state._seed)

    key, state._rng_key = jax.random.split(state._rng_key)

    return key
