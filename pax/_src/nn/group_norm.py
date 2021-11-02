"""GroupNorm module."""

# This file is an adaptation from
# https://raw.githubusercontent.com/deepmind/dm-haiku/main/haiku/_src/group_norm.py
# which is under Apache License, Version 2.0.


import collections
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from ..core import ParameterModule
from ..core.rng import KeyArray, next_rng_key

# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class GroupNorm(ParameterModule):
    r"""Group normalization module.

    This applies group normalization to the x. This involves splitting the
    channels into groups before calculating the mean and variance. The default
    behavior is to compute the mean and variance over the spatial dimensions and
    the grouped channels. The mean and variance will never be computed over the
    created groups axis.

    It transforms the input ``x`` into:

    .. math::

       \d{outputs} = \d{scale} \dfrac{x - \mu}{\sigma + \epsilon} + \d{offset}

    Where :math:`\mu` and :math:`\sigma` are respectively the mean and standard
    deviation of ``x``.

    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:

      - No ``scale``/``offset`` in which case ``create_*`` should be set to
        ``False`` and ``scale``/``offset`` aren't passed when the module is
        called.
      - Trainable ``scale``/``offset`` in which case create_* should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the scale/offset
        parameters.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.
    """

    scale: jnp.ndarray
    offset: jnp.ndarray

    def __init__(
        self,
        groups: int,
        num_channels: int,
        axis: Union[int, slice, Sequence[int]] = slice(1, None),
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
        scale_init: Optional[Callable] = None,
        offset_init: Optional[Callable] = None,
        data_format: str = "channels_last",
        *,
        rng_key: Optional[KeyArray] = None,
        name: Optional[str] = None,
    ):
        """Constructs a ``GroupNorm`` module.

        Args:
            groups: number of groups to divide the channels by. The number of channels
                must be divisible by this.
            num_channels: number of channels.
            axis: ``int``, ``slice`` or sequence of ints representing the axes which
                should be normalized across. By default this is all but the first
                dimension. For time series data use `slice(2, None)` to average over the
                none Batch and Time data.
            create_scale: whether to create a trainable scale per channel applied
                after the normalization.
            create_offset: whether to create a trainable offset per channel applied
                after normalization and scaling.
            eps: Small epsilon to add to the variance to avoid division by zero.
                Defaults to ``1e-5``.
            scale_init: Optional initializer for the scale parameter. Can only be set
                if ``create_scale=True``. By default scale is initialized to ``1``.
            offset_init: Optional initializer for the offset parameter. Can only be
                set if ``create_offset=True``. By default offset is initialized to
                ``0``.
            data_format: The data format of the input. Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default it is ``channels_last``.
            name: Name of the module.
        """
        super().__init__(name=name)

        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, collections.abc.Iterable) and all(
            isinstance(ax, int) for ax in axis
        ):
            self.axis = axis
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.groups = groups
        self.eps = eps
        self.data_format = data_format
        if data_format in ["channels_first", "NC..."]:
            self.channel_index = 1
        elif data_format in ["channels_last", "N...C"]:
            self.channel_index = -1
        else:
            raise ValueError(f"Data format `{data_format}` is not supported.")

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or jax.nn.initializers.ones
        self.offset_init = offset_init or jax.nn.initializers.zeros

        param_shape = [num_channels]
        rng_key = next_rng_key() if rng_key is None else rng_key
        rng1, rng2 = jax.random.split(rng_key)
        if create_scale:
            self.scale = self.scale_init(rng1, param_shape)
        if create_offset:
            self.offset = self.offset_init(rng2, param_shape)

    def __call__(
        self,
        x: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Returns normalized inputs.

        Args:
          x: An n-D tensor of the ``data_format`` specified in the constructor
            on which the transformation is performed.
          scale: A tensor up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``x``. This is the scale applied to the normalized
            x. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: A tensor up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``x``. This is the offset applied to the normalized
            ``x``. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
          An n-d tensor of the same shape as x that has been normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")

        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        channels = x.shape[self.channel_index]
        if channels % self.groups != 0:
            raise ValueError(
                "The number of channels must be divisible by the number of groups, "
                f"was channels={channels}, groups={self.groups}"
            )

        if self.create_scale:
            scale = self.scale

        if self.create_offset:
            offset = self.offset

        first_input_shape, group_shape, axis = self._compute_shape_and_axis(x, channels)

        x = x.reshape(group_shape)
        mean = jnp.mean(x, axis, keepdims=True)
        # TODO(tycai): Consider faster but less precise variance formulation.
        var = jnp.var(x, axis, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x.reshape(first_input_shape)

        if scale is not None:
            scale = jax.lax.broadcast_to_rank(scale, x.ndim)
            x = x * scale

        if offset is not None:
            offset = jax.lax.broadcast_to_rank(offset, x.ndim)
            x = x + offset

        return x

    def _compute_shape_and_axis(self, x: jnp.ndarray, channels: int):
        """Make it compatible with haiku code."""
        rank = x.ndim

        # Turns slice into list of axis
        if isinstance(self.axis, slice):
            axes = tuple(range(rank))
            axis = axes[self.axis]
        else:
            axis = self.axis

        if self.channel_index == -1:
            axis = tuple(a if a != rank - 1 else a + 1 for a in axis)
            group_shape = (-1,) + x.shape[1:-1] + (self.groups, channels // self.groups)
        else:
            assert self.channel_index == 1
            axis = tuple(a if a == 0 else a + 1 for a in axis)
            group_shape = (-1, self.groups, channels // self.groups) + x.shape[2:]

        first_input_shape = (-1,) + x.shape[1:]
        return first_input_shape, group_shape, axis
