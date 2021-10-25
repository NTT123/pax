"""Random Number Generator."""

from .threading_local import KeyArray, next_rng_key, seed_rng_key

__all__ = (
    "KeyArray",
    "next_rng_key",
    "seed_rng_key",
)
