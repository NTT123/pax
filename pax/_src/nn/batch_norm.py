"""BatchNorm modules."""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from ..core import Module, parameters_method
from .ema import EMA


class BatchNorm(Module):
    """A Generic BatchNorm Module.

    Normalize a mini-batch of data by subtracting its mean and dividing by its standard deviation.

    Use EMA modules to track the averaged mean and averaged variance for later uses in `eval` mode.
    """

    scale: Optional[jnp.ndarray]
    offset: Optional[jnp.ndarray]

    parameters = parameters_method("scale", "offset")

    ema_mean: EMA
    ema_var: EMA

    reduced_axes: Sequence[int]
    create_offset: bool
    create_scale: bool
    eps: float
    data_format: Optional[str]

    def __init__(
        self,
        num_channels: int,
        create_scale: bool = True,
        create_offset: bool = True,
        decay_rate: float = 0.9,
        eps: float = 1e-5,
        data_format: Optional[str] = None,
        reduced_axes=None,
        param_shape=None,
        *,
        name: Optional[str] = None,
    ):
        """Create a new BatchNorm module.

        Arguments:
            num_channels: the number of filters.
            create_scale: create a trainable `scale` parameter.
            create_offset: create a trainable `offset` parameter.
            decay_rate: the decay rate for tracking the averaged mean and the averaged variance.
            eps: a small positive number to avoid divided by zero.
            data_format:  the data format ["NHWC", NCHW", "NWC", "NCW"].
            reduced_axes: list of axes that will be reduced in the `jnp.mean` computation.
            param_shape: the shape of parameters.
        """
        super().__init__(name=name)
        assert 0 <= decay_rate <= 1

        self.num_channels = num_channels
        self.data_format = data_format
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.decay_rate = decay_rate

        self.reduced_axes = tuple(reduced_axes)

        if create_scale:
            self.scale = jnp.ones(param_shape, dtype=jnp.float32)
        else:
            self.scale = None
        if create_offset:
            self.offset = jnp.zeros(param_shape, dtype=jnp.float32)
        else:
            self.offset = None

        # initial values do not matter because debias=True
        initial_mean = jnp.zeros(param_shape, dtype=jnp.float32)
        self.ema_mean = EMA(initial_mean, decay_rate, debias=True)
        initial_var = jnp.ones(param_shape, dtype=jnp.float32)
        self.ema_var = EMA(initial_var, decay_rate, debias=True)

    def __call__(self, x):
        if self.training:
            batch_mean = jnp.mean(x, axis=self.reduced_axes, keepdims=True)
            batch_mean_of_squares = jnp.mean(
                jnp.square(x), axis=self.reduced_axes, keepdims=True
            )
            batch_var = batch_mean_of_squares - jnp.square(batch_mean)
            self.ema_mean(batch_mean)
            self.ema_var(batch_var)
        else:
            batch_mean = self.ema_mean.averages
            batch_var = self.ema_var.averages

        if self.create_scale:
            scale = self.scale
        else:
            scale = 1.0

        if self.create_offset:
            offset = self.offset
        else:
            offset = 0.0

        inv = scale * jax.lax.rsqrt(batch_var + self.eps)
        x = (x - batch_mean) * inv + offset
        return x

    def __repr__(self):
        info = {
            "num_channels": self.num_channels,
            "create_scale": self.create_scale,
            "create_offset": self.create_offset,
            "data_format": self.data_format,
            "decay_rate": self.decay_rate,
        }
        return self._repr(info)

    def summary(self, return_list: bool = False):
        lines = super().summary(return_list=True)
        if return_list:
            return lines[:1]
        else:
            return lines[0]


class BatchNorm1D(BatchNorm):
    """The 1D version of BatchNorm."""

    def __init__(
        self,
        num_channels: int,
        create_scale: bool = True,
        create_offset: bool = True,
        decay_rate: float = 0.9,
        eps: float = 1e-5,
        data_format: str = "NWC",
        *,
        name: Optional[str] = None,
    ):
        assert data_format in ["NWC", "NCW"], "expecting a correct `data_format`"

        param_shape = [1, 1, 1]
        if data_format == "NWC":
            axis = -1
            reduced_axes = [0, 1]
        else:
            axis = 1
            reduced_axes = [0, 2]
        param_shape[axis] = num_channels

        super().__init__(
            num_channels=num_channels,
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps,
            data_format=data_format,
            param_shape=param_shape,
            reduced_axes=reduced_axes,
            name=name,
        )


class BatchNorm2D(BatchNorm):
    """The 2D version of BatchNorm."""

    def __init__(
        self,
        num_channels: int,
        create_scale: bool = True,
        create_offset: bool = True,
        decay_rate: float = 0.9,
        eps: float = 1e-5,
        data_format: str = "NHWC",
        *,
        name: Optional[str] = None,
    ):
        assert data_format in ["NHWC", "NCHW"], "expecting a correct `data_format`"

        param_shape = [1, 1, 1, 1]
        if data_format == "NHWC":
            axis = -1
            reduced_axes = [0, 1, 2]
        else:
            axis = 1
            reduced_axes = [0, 2, 3]
        param_shape[axis] = num_channels

        super().__init__(
            num_channels=num_channels,
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps,
            data_format=data_format,
            param_shape=param_shape,
            reduced_axes=reduced_axes,
            name=name,
        )
