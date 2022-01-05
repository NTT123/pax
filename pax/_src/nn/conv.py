"""Conv modules."""

# Following the jax convolution tutorial:
# https://jax.readthedocs.io/en/latest/notebooks/convolutions.html
#

from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..core import ParameterModule
from ..core.rng import KeyArray, next_rng_key


class Conv(ParameterModule):
    """Convolution Base Class."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format=None,
        feature_group_count: int = 1,
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        assert out_features % feature_group_count == 0
        assert in_features > 0 and out_features > 0, "positive values"
        assert data_format in [
            "NWC",
            "NCW",
            "NHWC",
            "NCHW",
        ], f"Data format `{data_format}` is not supported."
        super().__init__(name=name)

        self.in_features = in_features
        self.out_features = out_features

        ndim = len(data_format) - 2

        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape,) * ndim
        self.kernel_shape = kernel_shape

        if isinstance(stride, int):
            stride = (stride,) * ndim
        self.stride = stride

        if isinstance(rate, int):
            rate = (rate,) * ndim
        self.rate = rate

        if isinstance(padding, str):
            assert padding in ["SAME", "VALID"], f"Not supported padding `{padding}`"
        elif isinstance(padding, tuple):
            raise ValueError(
                "Tuple type padding is not supported. Use `[ (int, int) ]` instead."
            )
        self.padding = padding
        self.with_bias = with_bias
        self.data_format = data_format
        self.feature_group_count = feature_group_count

        rng_key = next_rng_key() if rng_key is None else rng_key
        w_rng_key, b_rng_key = jax.random.split(rng_key)

        w_shape = [*kernel_shape, in_features // feature_group_count, out_features]
        if ndim == 1:
            self.kernel_format = "WIO"
        else:
            self.kernel_format = "HWIO"
        self.kernel_dilation = (1,) * ndim

        if w_init is None:
            fan_in = np.prod(w_shape[:-1])
            w_init = jax.nn.initializers.normal(stddev=1.0 / np.sqrt(fan_in))

        self.weight = w_init(w_rng_key, w_shape)

        if with_bias:
            if b_init is None:
                b_init = jax.nn.initializers.zeros
            b_shape = [out_features]
            self.bias = b_init(b_rng_key, b_shape)
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert len(x.shape) == len(self.kernel_format)
        n_channels = x.shape[1] if self.data_format[1] == "C" else x.shape[-1]
        if n_channels != self.in_features:
            raise ValueError(
                f"Expecting {self.in_features} input channels. Get {n_channels} channels."
            )

        dimension_numbers = jax.lax.conv_dimension_numbers(
            x.shape,
            self.weight.shape,
            (self.data_format, self.kernel_format, self.data_format),
        )

        x = jax.lax.conv_general_dilated(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.kernel_dilation,  # lhs/image dilation
            self.rate,  # rhs/kernel dilation
            dimension_numbers,
            feature_group_count=self.feature_group_count,
        )

        if self.bias is not None:
            if self.data_format == "NCHW":
                # pylint: disable=unsubscriptable-object
                x = x + self.bias[None, :, None, None]
            elif self.data_format == "NCW":
                # pylint: disable=unsubscriptable-object
                x = x + self.bias[None, :, None]
            elif self.data_format[-1] == "C":
                x = x + self.bias
            else:
                raise ValueError(
                    f"Not expecting this to happen, data_format {self.data_format}"
                )

        return x

    def __repr__(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "kernel_shape": self.kernel_shape,
            "padding": self.padding,
            "stride": self.stride,
            "rate": self.rate,
            "with_bias": self.with_bias,
            "data_format": self.data_format,
            "feature_group_count": self.feature_group_count,
        }
        all_one = lambda x: all(e == 1 for e in x)
        if all_one(info["rate"]):
            del info["rate"]
        if all_one(info["stride"]):
            del info["stride"]
        if info["feature_group_count"] == 1:
            del info["feature_group_count"]

        return self._repr(info)


class Conv1D(Conv):
    """1D Convolution Module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format: str = "NWC",
        feature_group_count: int = 1,
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        """Initializes the module.

        (Haiku documentation)

        Arguments:
            in_features: Number of input channels.
            out_features: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length 1.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length 1. Defaults to 1.
            rate: Optional kernel dilation rate. Either an integer or a sequence of
                length 1. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: Optional padding. Either ``VALID`` or ``SAME`` or
                sequence of `Tuple[int, int]` representing the padding before and after
                for each spatial dimension. Defaults to ``SAME``.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NWC`` or ``NCW``. By
                default, ``NWC``.
            feature_group_count: Optional number of groups in group convolution.
                Default value of 1 corresponds to normal dense convolution.
                If a higher value is used, convolutions are applied separately to that many groups,
                then stacked together.
            name: The name of the module.
            rng_key: The random key.
        """

        assert data_format in [
            "NWC",
            "NCW",
        ], f"Data format `{data_format}` is not supported. Use `NWC` or `NCW`."
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            feature_group_count=feature_group_count,
            name=name,
            rng_key=rng_key,
        )


class Conv2D(Conv):
    """2D Convolution Module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format: str = "NHWC",
        feature_group_count: int = 1,
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        """Initializes the module.

        (Haiku documentation)

        Arguments:
            in_features: Number of output channels.
            out_features: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length 2.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length 2. Defaults to 1.
            rate: Optional kernel dilation rate. Either an integer or a sequence of
                length 2. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: Optional padding. Either ``VALID`` or ``SAME`` or
                sequence of `Tuple[int, int]` representing the padding before and after
                for each spatial dimension. Defaults to ``SAME``.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
                default, ``NHWC``.
            feature_group_count: Optional number of groups in group convolution.
                Default value of 1 corresponds to normal dense convolution.
                If a higher value is used, convolutions are applied separately to that many groups,
                then stacked together.
            name: The name of the module.
            rng_key: The random key.
        """

        assert data_format in [
            "NHWC",
            "NCHW",
        ], f"Data format `{data_format}` is not supported. Use `NHWC` or `NCHW`."
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            feature_group_count=feature_group_count,
            name=name,
            rng_key=rng_key,
        )


class ConvTranspose(ParameterModule):
    """Convolution Transpose Base Class."""

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format=None,
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        assert in_features > 0 and out_features > 0, "positive values"
        assert data_format in [
            "NWC",
            "NCW",
            "NHWC",
            "NCHW",
        ], f"Data format `{data_format}` is not supported."
        super().__init__(name=name)

        self.in_features = in_features
        self.out_features = out_features

        ndim = len(data_format) - 2

        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape,) * ndim
        self.kernel_shape = kernel_shape

        if isinstance(stride, int):
            stride = (stride,) * ndim
        self.stride = stride

        if isinstance(padding, str):
            assert padding in ["SAME", "VALID"], f"Not supported padding `{padding}`"
        self.padding = padding

        self.with_bias = with_bias
        self.data_format = data_format

        rng_key = next_rng_key() if rng_key is None else rng_key
        w_rng_key, b_rng_key = jax.random.split(rng_key)

        w_shape = [*kernel_shape, out_features, in_features]
        if ndim == 1:
            self.kernel_format = "WOI"
        else:
            self.kernel_format = "HWOI"
        self.kernel_dilation = (1,) * ndim

        if w_init is None:
            fan_in = np.prod(w_shape[:-2] + [in_features])
            w_init = jax.nn.initializers.normal(stddev=1.0 / np.sqrt(fan_in))
        if b_init is None:
            b_init = jax.nn.initializers.zeros

        self.weight = w_init(w_rng_key, w_shape)
        b_shape = [out_features]
        self.bias = b_init(b_rng_key, b_shape)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert len(x.shape) == len(self.kernel_format)

        dimension_numbers = jax.lax.conv_dimension_numbers(
            x.shape,
            self.weight.shape,
            (self.data_format, self.kernel_format, self.data_format),
        )

        x = jax.lax.conv_transpose(
            lhs=x,
            rhs=self.weight,
            strides=self.stride,
            padding=self.padding,
            dimension_numbers=dimension_numbers,
        )

        if self.with_bias:
            if self.data_format == "NCHW":
                x = x + self.bias[None, :, None, None]
            elif self.data_format == "NCW":
                x = x + self.bias[None, :, None]
            elif self.data_format[-1] == "C":
                x = x + self.bias
            else:
                raise ValueError(
                    f"Not expecting this to happen, data_format={self.data_format}"
                )

        return x

    def __repr__(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "data_format": self.data_format,
            "padding": self.padding,
            "stride": self.stride,
            "with_bias": self.with_bias,
        }
        return self._repr(info)


class Conv1DTranspose(ConvTranspose):
    """1D Convolution Transpose Module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format: str = "NWC",
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        """Initializes the module.

        (Haiku documentation)


        Arguments:
            in_features: Number of input channels.
            out_features: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length 1.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length 1. Defaults to 1.
            padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
                Defaults to ``SAME``.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NWC`` or ``NCW``. By
                default, ``NWC``.
            name: The name of the module.
            rng_key: The random key.
        """

        assert data_format in [
            "NWC",
            "NCW",
        ], f"Data format `{data_format}` is not supported. Use `NWC` or `NCW`."

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            name=name,
            rng_key=rng_key,
        )


class Conv2DTranspose(ConvTranspose):
    """2D Convolution Transpose Module."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        data_format: str = "NHWC",
        *,
        name: Optional[str] = None,
        rng_key: Optional[KeyArray] = None,
    ):
        """Initializes the module.

        (Haiku documentation)


        Arguments:
            in_features: Number of input channels.
            out_features: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length 1.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length 1. Defaults to 1.
            padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
                Defaults to ``SAME``.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NHWC`` or ``NHCW``. By
                default, ``NHWC``.
            name: The name of the module.
            rng_key: The random key.
        """

        assert data_format in [
            "NHWC",
            "NCHW",
        ], f"Data format `{data_format}` is not supported. Use `NHWC` or `NCHW`."

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            name=name,
            rng_key=rng_key,
        )
