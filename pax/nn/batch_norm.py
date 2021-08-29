"""BatchNorm Module."""
from typing import Optional, Sequence, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np

from ..haiku import HaikuParam, HaikuState
from ..module import Module
from ..rng import next_rng_key


class BatchNorm(Module):
    """BatchNorm proxy."""

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
        rng_key: Optional[jnp.ndarray] = None
    ):
        """
        Arguments:
            input_shape: The shape of input tensor. For example `[None, None, 3]`. Use `None` to indicate unknown value.
        """
        super().__init__()

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
        x = np.empty([(1 if i is None else i) for i in input_shape], dtype=np.float32)
        self.params, self.state = self.fwd.init(rng_key, x, is_training=self.training)
        self.register_parameter_subtree(
            "params", hk.data_structures.to_mutable_dict(self.params)
        )
        self.register_state_subtree(
            "state", hk.data_structures.to_mutable_dict(self.state)
        )

    def __call__(self, x):
        x, state = self.fwd.apply(self.params, self.state, x, is_training=self.training)
        if self.training:
            # TODO: remove this
            self.state = hk.data_structures.to_mutable_dict(state)
        return x
