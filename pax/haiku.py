"""Convert Haiku module to pax.Module"""
import logging
from typing import Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from haiku import LSTMState, dropout, dynamic_unroll
from pax.module import Module

HaikuState = Dict[str, Dict[str, jnp.ndarray]]
HaikuParam = Dict[str, Dict[str, jnp.ndarray]]


def from_haiku(
    cls,
    use_rng: bool = False,
    delay: bool = False,
    pass_is_training: bool = False,
):

    """Build a pax.Module class from haiku cls.

    Arguments:
        cls: dm-haiku class. For example, hk.BatchNorm.
        use_rng: generate a new rng key for each call.
        delay: bool, delay the initialization process until the module is exectied.
        pass_is_training: pass `is_training` as an argument to module. `hk.BatchNorm` needs this.

    Returns:
        haiku_module_builder, a function which creates Pax Module.
    """

    def haiku_module_builder(*args, **kwargs):
        fwd = lambda *u, **v: cls(*args, **kwargs)(*u, **v)
        hk_fwd = hk.transform_with_state(fwd)

        class HaikuModule(Module):
            params: HaikuParam
            state: HaikuState
            rng_key: jnp.ndarray
            _is_haiku_initialized: bool = False

            def init_haiku_module(self, u, v):
                params, state = map(
                    hk.data_structures.to_mutable_dict,
                    hk_fwd.init(self.rng_key, *u, **v),
                )
                self.register_parameter_subtree("params", params)
                self.register_state_subtree("state", state)
                self.register_state_subtree("rng_key", self.rng_key)
                self._is_haiku_initialized = True

            def __init__(self, *u, rng_key: Optional[jnp.ndarray] = None, **v) -> None:
                super().__init__()
                self.rng_key = jax.random.PRNGKey(42) if rng_key is None else rng_key
                if pass_is_training:
                    v["is_training"] = self.training

                if delay == False:
                    self.init_haiku_module(u, v)

            def __call__(self, *args, **kwargs):
                if not self._is_haiku_initialized:
                    logging.warning(
                        "Initialize a haiku module on the fly! "
                        "Make sure you're doing this right after a module is created. "
                        "Or at least, before `self.parameters()` method is called."
                    )
                    self.init_haiku_module(args, kwargs)

                if use_rng:
                    new_rng_key, rng_key = jax.random.split(self.rng_key, 2)
                else:
                    rng_key = None
                if pass_is_training:
                    kwargs["is_training"] = self.training
                out, state = hk_fwd.apply(
                    self.params, self.state, rng_key, *args, **kwargs
                )
                if self.training:
                    # only update state in training mode.
                    if use_rng:
                        self.rng_key = new_rng_key
                    self.state = hk.data_structures.to_mutable_dict(state)
                return out

        return HaikuModule

    return haiku_module_builder


def batch_norm_2d(
    num_channels: int, axis: int = -1, decay_rate=0.99, cross_replica_axis=None
):
    """Return a BatchNorm module."""
    BatchNorm = from_haiku(hk.BatchNorm, pass_is_training=True)(
        create_scale=True,
        create_offset=True,
        decay_rate=decay_rate,
        cross_replica_axis=cross_replica_axis,
    )
    shape = [1, 1, 1, 1]
    shape[axis] = num_channels
    x = np.empty((num_channels,), dtype=np.float32).reshape(shape)
    return BatchNorm(x)


def layer_norm(num_channels: int, axis: int = -1):
    LayerNorm = from_haiku(hk.LayerNorm)(
        axis=axis, create_scale=True, create_offset=True
    )
    shape = [1, 1, 1, 1]
    shape[axis] = num_channels
    x = np.empty((num_channels,), dtype=np.float32).reshape(shape)
    return LayerNorm(x)


def linear(in_dim: int, out_dim: int, with_bias: bool = True):
    Linear = from_haiku(hk.Linear)(output_size=out_dim, with_bias=with_bias)
    shape = (1, in_dim)
    x = np.empty(shape, dtype=np.float32)
    return Linear(x)


def lstm(hidden_dim: int):
    LSTM = from_haiku(hk.LSTM)(hidden_size=hidden_dim)

    def initial_state(o, batch_size):
        h0 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        c0 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        return LSTMState(h0, c0)

    LSTM.initial_state = initial_state
    x = np.empty((1, hidden_dim), dtype=np.float32)
    return LSTM(x, LSTM.initial_state(LSTM, 1))


def gru(hidden_dim: int):
    GRU = from_haiku(hk.GRU)(hidden_size=hidden_dim)

    def initial_state(o, batch_size):
        h0 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        return h0

    GRU.initial_state = initial_state
    x = np.empty((1, hidden_dim), dtype=np.float32)
    return GRU(x, GRU.initial_state(GRU, 1))


def embed(vocab_size: int, embed_dim: int):
    Embed = from_haiku(hk.Embed)(vocab_size=vocab_size, embed_dim=embed_dim)
    x = np.empty((1, 1), dtype=np.int32)
    return Embed(x)


def conv_1d(
    input_channels: int,
    output_channels: int,
    kernel_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    rate: Union[int, Sequence[int]] = 1,
    padding: str = "SAME",
    with_bias: bool = True,
    w_init=None,
    b_init=None,
    data_format="NWC",
    feature_group_count=1,
):
    Conv1D = from_haiku(hk.Conv1D)(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        feature_group_count=feature_group_count,
    )
    x = np.empty((input_channels), dtype=np.float32)
    shape = [1, 1, 1]
    assert data_format in ["NWC", "NCW"]
    if data_format == "NWC":
        shape[-1] = input_channels
    elif data_format == "NCW":
        shape[1] = input_channels
    x = np.reshape(x, shape)
    return Conv1D(x)


def conv_2d(
    input_channels: int,
    output_channels: int,
    kernel_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    rate: Union[int, Sequence[int]] = 1,
    padding: str = "SAME",
    with_bias: bool = True,
    w_init=None,
    b_init=None,
    data_format="NHWC",
    feature_group_count=1,
):
    Conv2D = from_haiku(hk.Conv2D)(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        feature_group_count=feature_group_count,
    )
    assert data_format in ["NHWC", "NCHW"]
    x = np.empty((input_channels), dtype=np.float32)
    shape = [1, 1, 1, 1]
    if data_format == "NHWC":
        shape[-1] = input_channels
    elif data_format == "NCHW":
        shape[1] = input_channels
    x = np.reshape(x, shape)
    return Conv2D(x)


def conv_1d_transpose(
    input_channels: int,
    output_channels: int,
    kernel_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    output_shape=None,
    padding: str = "SAME",
    with_bias: bool = True,
    w_init=None,
    b_init=None,
    data_format="NWC",
    mask=None,
):
    Conv1DTranspose = from_haiku(hk.Conv1DTranspose)(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        output_shape=output_shape,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
    )
    assert data_format in ["NWC", "NCW"]
    x = np.empty((input_channels), dtype=np.float32)
    shape = [1, 1, 1]
    if data_format == "NWC":
        shape[-1] = input_channels
    elif data_format == "NCW":
        shape[1] = input_channels
    x = np.reshape(x, shape)
    return Conv1DTranspose(x)


def conv_2d_transpose(
    input_channels: int,
    output_channels: int,
    kernel_shape: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]] = 1,
    output_shape=None,
    padding: str = "SAME",
    with_bias: bool = True,
    w_init=None,
    b_init=None,
    data_format="NHWC",
    mask=None,
):
    Conv2DTranspose = from_haiku(hk.Conv2DTranspose)(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        output_shape=output_shape,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
    )
    assert data_format in ["NHWC", "NCHW"]
    x = np.empty((input_channels), dtype=np.float32)
    shape = [1, 1, 1, 1]
    if data_format == "NHWC":
        shape[-1] = input_channels
    elif data_format == "NCHW":
        shape[1] = input_channels
    x = np.reshape(x, shape)
    return Conv2DTranspose(x)


def avg_pool(window_shape, strides, padding, channel_axis=-1):
    AvgPool = from_haiku(hk.AvgPool)(
        window_shape=window_shape,
        strides=strides,
        padding=padding,
        channel_axis=channel_axis,
    )
    return lambda x: AvgPool(x)(x)


def max_pool(window_shape, strides, padding, channel_axis=-1):
    MaxPool = from_haiku(hk.MaxPool)(
        window_shape=window_shape,
        strides=strides,
        padding=padding,
        channel_axis=channel_axis,
    )
    return lambda x: MaxPool(x)(x)
