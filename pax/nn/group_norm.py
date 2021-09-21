from typing import Optional

import jax.numpy as jnp

from .. import initializers
from ..module import Module
from ..rng import KeyArray
from .layer_norm import LayerNorm


class GroupNorm(Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        axis=-1,
        eps=1e-5,
        scale_init: Optional[initializers.Initializer] = None,
        offset_init: Optional[initializers.Initializer] = None,
        use_fast_variance: bool = False,
        *,
        rng_key: Optional[KeyArray] = None,
        name: Optional[str] = None,
    ):
        """Constructs a GroupNorm module.

        Arguments:
            num_groups: Interger, number of groups.
            num_channels: Interger, size of the last dimension.
            axis: Integer, list of integers, or slice indicating which axes to normalize over.
            eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``, as in the paper and Sonnet.
            scale_init: Optional initializer for gain (aka scale). By default, one.
            offset_init: Optional initializer for bias (aka offset). By default, zero.
            use_fast_variance: If true, use a faster but less numerically stable formulation for computing variance.
            rng_key: RNG key.
            name: module name.
        """
        super().__init__(name=name)
        assert axis == [-1, 1], "Only support axis=-1 or axis=1"

        assert (
            num_channels % num_groups == 0
        ), "Number of channels must be divisible by number of groups"
        self.layer_norm = LayerNorm(
            num_channels // num_groups,
            axis=axis,
            create_scale=True,
            create_offset=True,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            use_fast_variance=use_fast_variance,
            rng_key=rng_key,
        )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.axis = axis

    def __call__(self, x):
        """Normalize inputs by groups of channels."""
        assert x.shape[self.axis] == self.num_channels
        original_shape = x.shape
        n_channels = self.num_channels // self.num_groups

        if self.axis == -1:
            shape = x.shape[:-1] + (self.num_groups, n_channels)
        elif self.axis == 1:
            shape = (x.shape[0] * self.num_groups, n_channels) + x.shape[2:]
        else:
            raise ValueError("Impossible")

        x = jnp.reshape(x, shape)
        x = self.layer_norm(x)
        x = jnp.reshape(x, original_shape)
        return x
