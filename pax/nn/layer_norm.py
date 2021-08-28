"""LayerNorm Module.

The implementation is almost identical to dm-haiku LayerNorm at: 
https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/layer_norm.py
deepmind/dm-haiku is licensed under the Apache License 2.0

Differences:
    1. We need to input ``num_channels``, the size of the last dimension, to initialize scale/offset parameters.
    2. We can input `rng_key` to seed the value of scale/offset parameters.
"""
import collections
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from haiku import initializers

from .. import initializers
from ..module import Module, Union
from ..rng import next_rng_key
from ..tree import Parameter


class LayerNorm(Module):
    """LayerNorm module.
    See: https://arxiv.org/abs/1607.06450.
    """

    scale: Optional[Parameter] = None
    offset: Optional[Parameter] = None

    def __init__(
        self,
        num_channels: int,
        axis: Union[int, Sequence[int], slice],
        create_scale: bool,
        create_offset: bool,
        eps: float = 1e-5,
        scale_init: Optional[initializers.Initializer] = None,
        offset_init: Optional[initializers.Initializer] = None,
        use_fast_variance: bool = False,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        """Constructs a LayerNorm module.
        Args:
          num_channels: Interger, size of the last dimension. The data format is ``[N, ..., C]``.
          axis: Integer, list of integers, or slice indicating which axes to
            normalize over.
          create_scale: Bool, defines whether to create a trainable scale
            per channel applied after the normalization.
          create_offset: Bool, defines whether to create a trainable offset
            per channel applied after normalization and scaling.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). By default, one.
          offset_init: Optional initializer for bias (aka offset). By default, zero.
          use_fast_variance: If true, use a faster but less numerically stable
            formulation for computing variance.
          rng_key: RNG key.
        """
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, collections.abc.Iterable) and all(
            isinstance(ax, int) for ax in axis
        ):
            self.axis = tuple(axis)
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.eps = eps
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or initializers.ones
        self.offset_init = offset_init or initializers.zeros
        self.use_fast_variance = use_fast_variance

        param_shape = [num_channels]
        if rng_key is None:
            rng_key = next_rng_key()
        rng1, rng2 = jax.random.split(rng_key)
        if create_scale:
            self.scale = self.scale_init(param_shape, jnp.float32, rng1)
        if create_offset:
            self.offset = self.offset_init(param_shape, jnp.float32, rng2)

    def __call__(
        self,
        inputs: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Connects the layer norm.
        Args:
          inputs: An array, where the data format is ``[N, ..., C]``.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        axis = self.axis
        if isinstance(axis, slice):
            axis = tuple(range(inputs.ndim)[axis])

        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        if self.use_fast_variance:
            mean_of_squares = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
            variance = mean_of_squares - jnp.square(mean)
        else:
            variance = jnp.var(inputs, axis=axis, keepdims=True)

        # param_shape = inputs.shape[-1:]
        if self.create_scale:
            scale = self.scale
        elif scale is None:
            scale = np.array(1.0, dtype=inputs.dtype)

        if self.create_offset:
            offset = self.offset
        elif offset is None:
            offset = np.array(0.0, dtype=inputs.dtype)

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + offset
