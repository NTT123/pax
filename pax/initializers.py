from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp

import haiku as hk

from .rng import next_rng_key

Initializer = Callable[[Sequence[int], Any, Optional[jnp.ndarray]], jnp.ndarray]


def from_haiku_initializer(fn: hk.initializers.Initializer) -> Initializer:
    """Convert haiku initializer to pax initializer."""

    def _fn(shape: Sequence[int], dtype: Any, rng_key: Optional[jnp.ndarray] = None):
        rng_key = next_rng_key() if rng_key is None else rng_key
        return hk.transform(fn).apply({}, rng_key, shape, dtype)

    return _fn


def zeros(shape: Sequence[int], dtype: Any, rng_key=None):
    """Initialize all zeros."""
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: Any, rng_key=None):
    """Initialize all ones."""
    return jnp.ones(shape, dtype=dtype)


def truncated_normal(stddev: float = 1.0, mean: float = 0.0):
    """Initialize from truncated normal distribution."""

    def _truncated_normal_init(shape, dtype, rng_key):
        noise = jax.random.truncated_normal(
            key=rng_key, shape=shape, dtype=dtype, lower=-2.0, upper=2.0
        )
        return noise * stddev + mean

    return _truncated_normal_init


def random_normal(stddev: float = 1.0, mean: float = 0.0):
    """Initialize from normal distribution."""

    def _random_normal_init(shape, dtype, rng_key):
        noise = jax.random.normal(key=rng_key, shape=shape, dtype=dtype)
        return noise * stddev + mean

    return _random_normal_init


def variance_scaling(
    scale: float = 1, mode: str = "fan_in", distribution: str = "truncated_normal"
):
    """Initializer which adapts its scale to the shape of the initialized array.

    (Haiku documentation)

    The initializer first computes the scaling factor ``s = scale / n``, where n
    is:

        - Number of input units in the weight tensor, if ``mode = fan_in``.
        - Number of output units, if ``mode = fan_out``.
        - Average of the numbers of input and output units, if ``mode = fan_avg``.

    Then, with ``distribution="truncated_normal"`` or ``"normal"``,
    samples are drawn from a distribution with a mean of zero and a standard
    deviation (after truncation, if used) ``stddev = sqrt(s)``.
    With ``distribution=uniform``, samples are drawn from a uniform distribution
    within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.
    The variance scaling initializer can be configured to generate other standard
    initializers using the scale, mode and distribution arguments. Here are some
    example configurations:

    ==============  ==============================================================
    Name            Parameters
    ==============  ==============================================================
    glorot_uniform  variance_scaling(1.0, "fan_avg", "uniform")
    glorot_normal   variance_scaling(1.0, "fan_avg", "truncated_normal")
    lecun_uniform   variance_scaling(1.0, "fan_in",  "uniform")
    lecun_normal    variance_scaling(1.0, "fan_in",  "truncated_normal")
    he_uniform      variance_scaling(2.0, "fan_in",  "uniform")
    he_normal       variance_scaling(2.0, "fan_in",  "truncated_normal")
    ==============  ==============================================================
    """

    return from_haiku_initializer(
        hk.initializers.VarianceScaling(
            scale=scale, mode=mode, distribution=distribution
        )
    )
