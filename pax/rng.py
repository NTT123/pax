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


def seed_rng_key(seed: int = None):
    """Seed state.rng_key with [0, seed]"""
    if seed is None:
        seed = 42
        logging.warn(
            f"Seeding RNG key with seed {seed}. Use `pax.seed_rng_key` function to avoid this warning."
        )
    state._rng_key = jax.random.PRNGKey(seed)


def next_rng_key() -> jnp.ndarray:
    """Return a random RNG key."""
    if state._rng_key is None:
        seed_rng_key()
    key, state._rng_key = jax.random.split(state._rng_key)

    return key
