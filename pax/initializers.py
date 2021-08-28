from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp

import haiku as hk

from .rng import next_rng_key

Initializer = Callable[[Sequence[int], Any, Optional[jnp.ndarray]], jnp.ndarray]


def from_haiku_initializer(fn: hk.initializers.Initializer) -> Initializer:
    def _fn(shape: Sequence[int], dtype: Any, rng_key: Optional[jnp.ndarray] = None):
        rng_key = next_rng_key() if rng_key is None else rng_key
        return hk.transform(fn).apply({}, rng_key, shape, dtype)

    return _fn


def zeros(shape: Sequence[int], dtype: Any, rng_key=None):
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: Any, rng_key=None):
    return jnp.ones(shape, dtype=dtype)


def truncated_normal(stddev: float = 1.0, mean: float = 0.0):
    return from_haiku_initializer(
        hk.initializers.TruncatedNormal(stddev=stddev, mean=mean)
    )


def random_normal(stddev: float = 1.0, mean: float = 0.0):
    return from_haiku_initializer(
        hk.initializers.RandomNormal(stddev=stddev, mean=mean)
    )


def variance_scaling(
    scale: float = 1, mode: str = "fan_in", distribution: str = "truncated_normal"
):
    return from_haiku_initializer(
        hk.initializers.VarianceScaling(
            scale=scale, mode=mode, distribution=distribution
        )
    )
