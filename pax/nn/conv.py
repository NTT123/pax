""""Convolution Module"""

from typing import Optional, Sequence, Tuple, Union

import haiku as hk
import jax.numpy as jnp
import numpy as np

from ..module import Module
from ..rng import next_rng_key


class Conv1D(Module):
    """A proxy for dm-haiku Conv1D."""

    w: jnp.ndarray
    b: jnp.ndarray

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,  # TODO: remove hk
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: Optional[str] = "NWC",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        *,
        rng_key: jnp.ndarray = None,
    ):
        """
        Arguments:
            in_features: the number of input features.
            out_features: the number of output features.
            kernel_shape: convolution kernel shape.
            rng_key: the random key for initialization.
        """
        assert data_format in [
            "NCW",
            "NWC",
        ], f"data format {data_format} is not supported."
        self.in_features = in_features
        self.out_features = out_features

        def fwd(x):
            return hk.Conv1D(
                output_channels=out_features,
                kernel_shape=kernel_shape,
                stride=stride,
                rate=rate,
                padding=padding,
                with_bias=with_bias,
                w_init=w_init,
                b_init=b_init,
                data_format=data_format,
                mask=mask,
                feature_group_count=feature_group_count,
            )(x)

        self.fwd = hk.without_apply_rng(hk.transform(fwd))

        rng_key = next_rng_key() if rng_key is None else rng_key
        if data_format == "NCW":
            x = np.empty(shape=(1, in_features, 1), dtype=jnp.float32)
        elif data_format == "NWC":
            x = np.empty(shape=(1, 1, in_features), dtype=jnp.float32)
        params = self.fwd.init(rng_key, x)
        self.register_parameter("w", params["conv1_d"]["w"])
        self.register_parameter("b", params["conv1_d"]["b"] if with_bias else None)

    def __call__(self, x):
        return self.fwd.apply({"conv1_d": {"w": self.w, "b": self.b}}, x)


class Conv2D(Module):
    """A proxy for dm-haiku Conv2D."""

    w: jnp.ndarray
    b: jnp.ndarray

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: Optional[str] = "NHWC",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        *,
        rng_key: jnp.ndarray = None,
    ):
        """
        Arguments:
            in_features: the number of input features.
            out_features: the number of output features.
            kernel_shape: convolution kernel shape.
            rng_key: the random key for initialization.
        """
        assert data_format in [
            "NCHW",
            "NHWC",
        ], f"data format {data_format} is not supported."
        self.in_features = in_features
        self.out_features = out_features

        def fwd(x):
            return hk.Conv2D(
                output_channels=out_features,
                kernel_shape=kernel_shape,
                stride=stride,
                rate=rate,
                padding=padding,
                with_bias=with_bias,
                w_init=w_init,
                b_init=b_init,
                data_format=data_format,
                mask=mask,
                feature_group_count=feature_group_count,
            )(x)

        self.fwd = hk.without_apply_rng(hk.transform(fwd))

        rng_key = rng_key or next_rng_key()
        if data_format == "NCHW":
            x = np.empty(shape=(1, in_features, 1, 1), dtype=jnp.float32)
        elif data_format == "NHWC":
            x = np.empty(shape=(1, 1, 1, in_features), dtype=jnp.float32)
        params = self.fwd.init(rng_key, x)
        self.register_parameter("w", params["conv2_d"]["w"])
        self.register_parameter("b", params["conv2_d"]["b"] if with_bias else None)

    def __call__(self, x):
        return self.fwd.apply({"conv2_d": {"w": self.w, "b": self.b}}, x)
