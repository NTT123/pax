import jax.numpy as jnp

from ..module import Module
from .layer_norm import LayerNorm


class GroupNorm(Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        axis=-1,
        eps=1e-5,
        create_scale=True,
        create_offset=True,
    ):
        super().__init__()
        assert axis == -1, "Only support axis=-1"

        assert (
            num_channels % num_groups == 0
        ), "Number of channels must be divisible by number of groups"
        self.layer_norm = LayerNorm(
            num_channels // num_groups,
            axis=axis,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
        )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.axis = axis

    def __call__(self, x):
        assert x.shape[self.axis] == self.num_channels
        N, H, W, C = x.shape
        x = jnp.reshape(
            x, (N, H, W, self.num_groups, self.num_channels // self.num_groups)
        )
        x = self.layer_norm(x)
        x = jnp.reshape(x, (N, H, W, C))
        return x
