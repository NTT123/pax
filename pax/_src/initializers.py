"""Initializers"""

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .core.rng import KeyArray

Initializer = Callable[[Sequence[int], Any, KeyArray], jnp.ndarray]

# source:
# https://github.com/deepmind/dm-haiku/blob/48f5d5d9b7faabffb3860900c633229bc57e01df/haiku/_src/initializers.py#L34
def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        # Assuming convolution kernels (2D, 3D, or more.)
        # kernel_shape: (..., input_depth, depth)
        # TODO: this is not true for conv_tranpose.
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def zeros(shape: Sequence[int], dtype: Any, rng_key=None):
    """Initialize all zeros."""
    del rng_key
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Sequence[int], dtype: Any, rng_key=None):
    """Initialize all ones."""
    del rng_key
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


def random_uniform(minval=0.0, maxval=1.0):
    """Initialize from uniform distribution."""

    def _random_uniform(shape, dtype, rng_key):
        return jax.random.uniform(
            rng_key, shape=shape, dtype=dtype, minval=minval, maxval=maxval
        )

    return _random_uniform


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

    def _variance_scaling_init(shape, dtype, rng_key):
        fan_in, fan_out = _compute_fans(shape)

        if mode == "fan_in":
            scale_ = scale / max(1.0, fan_in)
        elif mode == "fan_out":
            scale_ = scale / max(1.0, fan_out)
        else:
            scale_ = scale / max(1.0, (fan_in + fan_out) / 2.0)

        if distribution == "truncated_normal":
            stddev = np.sqrt(scale_)
            distribution_stddev = np.asarray(0.87962566103423978, dtype=np.float32)
            stddev = stddev / distribution_stddev
            return truncated_normal(stddev=stddev)(shape, dtype, rng_key)
        elif distribution == "normal":
            stddev = np.sqrt(scale_)
            return random_normal(stddev=stddev)(shape, dtype, rng_key)
        else:
            limit = np.sqrt(3.0 * scale_)
            return random_uniform(minval=-limit, maxval=limit)(shape, dtype, rng_key)

    return _variance_scaling_init
