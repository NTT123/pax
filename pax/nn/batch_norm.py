"""BatchNorm Module."""
from typing import Optional, Sequence, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np

from ..haiku import HaikuParam, HaikuState
from ..module import Module
from ..rng import next_rng_key


class BatchNorm(Module):
    """BatchNorm Module."""

    params: HaikuParam = None
    state: HaikuState = None

    def __init__(
        self,
        input_shape: Sequence[Union[int, None]],
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 0.00001,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        data_format: str = "channels_last",
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        """Create a new BatchNorm module.

        Arguments:
            input_shape: The shape of input tensor. For example ``[None, None, 3]``. Use ``None`` to indicate unknown value.
            create_scale: Whether to include a trainable scaling factor.
            create_offset: Whether to include a trainable offset.
            decay_rate: Decay rate for EMA.
            eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
                as in the paper and Sonnet.
            scale_init: Optional initializer for gain (aka scale). Can only be set
                if ``create_scale=True``. By default, ``1``.
            offset_init: Optional initializer for bias (aka offset). Can only be set
                if ``create_offset=True``. By default, ``0``.
            axis: Which axes to reduce over. The default (``None``) signifies that all
                but the channel axis should be normalized. Otherwise this is a list of
                axis indices which will have normalization statistics calculated.
            cross_replica_axis: If not ``None``, it should be a string representing
                the axis name over which this module is being run within a ``jax.pmap``.
                Supplying this argument means that batch statistics are calculated
                across all replicas on that axis.
            data_format: The data format of the input. Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default it is ``channels_last``.
        """
        assert data_format in ["channels_first", "channels_last", "N...C", "NC..."]
        super().__init__(name=name)

        def fwd(x, is_training: bool):
            return hk.BatchNorm(
                create_scale=create_scale,
                create_offset=create_offset,
                decay_rate=decay_rate,
                eps=eps,
                scale_init=scale_init,
                offset_init=offset_init,
                axis=axis,
                cross_replica_axis=cross_replica_axis,
                data_format=data_format,
            )(x, is_training=is_training)

        self.fwd = hk.without_apply_rng(hk.transform_with_state(fwd))
        rng_key = next_rng_key() if rng_key is None else rng_key
        x = np.ones([(1 if i is None else i) for i in input_shape], dtype=np.float32)
        params, state = self.fwd.init(rng_key, x, is_training=self.training)
        self.register_parameter_subtree(
            "params", hk.data_structures.to_mutable_dict(params)
        )
        self.register_state_subtree("state", hk.data_structures.to_mutable_dict(state))

        if data_format in ["channels_last", "N...C"]:
            num_channels = input_shape[-1]
        else:
            num_channels = input_shape[1]

        self.info = {
            "num_channels": num_channels,
            "create_scale": create_scale,
            "create_offset": create_offset,
            "decay_rate": decay_rate,
            "data_format": data_format,
            "axis": axis,
            "cross_replica_axis": cross_replica_axis,
        }

    def __call__(self, x):
        x, state = self.fwd.apply(self.params, self.state, x, is_training=self.training)
        if self.training:
            # TODO: remove this
            self.state = hk.data_structures.to_mutable_dict(state)
        return x

    def __repr__(self):
        options = [f"{k}={v}" for (k, v) in self.info.items() if v is not None]
        options = ", ".join(options)
        name = f"({self.name}) " if self.name is not None else ""
        return f"{name}{self.__class__.__name__}[{options}]"


class BatchNorm1D(BatchNorm):
    """BatchNorm1D Module."""

    params: HaikuParam = None
    state: HaikuState = None

    def __init__(
        self,
        num_channels: int,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 0.00001,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        data_format: str = "channels_last",
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        shape = [1, 1, 1]
        if data_format in ["channels_last", "N...C"]:
            shape[-1] = num_channels
        else:
            shape[1] = num_channels

        super().__init__(
            input_shape=shape,
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            axis=axis,
            cross_replica_axis=cross_replica_axis,
            data_format=data_format,
            name=name,
            rng_key=rng_key,
        )


class BatchNorm2D(BatchNorm):
    """BatchNorm2D Module."""

    params: HaikuParam = None
    state: HaikuState = None

    def __init__(
        self,
        num_channels: int,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 0.00001,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        axis: Optional[Sequence[int]] = None,
        cross_replica_axis: Optional[str] = None,
        data_format: str = "channels_last",
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        shape = [1, 1, 1, 1]
        if data_format in ["channels_last", "N...C"]:
            shape[-1] = num_channels
        else:
            shape[1] = num_channels

        super().__init__(
            input_shape=shape,
            create_scale=create_scale,
            create_offset=create_offset,
            decay_rate=decay_rate,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            axis=axis,
            cross_replica_axis=cross_replica_axis,
            data_format=data_format,
            name=name,
            rng_key=rng_key,
        )
