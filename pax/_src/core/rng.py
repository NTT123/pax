"""Random Number Generator."""

from typing import Any, Union

import jax.numpy as jnp

from .threading_local import KeyArray, next_rng_key, seed_rng_key
