# Following the jax convolution tutorial:
# https://jax.readthedocs.io/en/latest/notebooks/convolutions.html
#

from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from .. import initializers
from ..module import Module
from ..rng import next_rng_key


class Conv(Module):

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[initializers.Initializer] = None,
        b_init: Optional[initializers.Initializer] = None,
        data_format=None,
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
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

        rng_key = next_rng_key() if rng_key is None else rng_key
        w_rng_key, b_rng_key = jax.random.split(rng_key)

        if w_init is None:
            w_init = initializers.truncated_normal()
        if b_init is None:
            b_init = initializers.zeros

        w_shape = [*kernel_shape, in_features, out_features]
        if ndim == 1:
            self.kernel_format = "WIO"
        else:
            self.kernel_format = "HWIO"
        self.kernel_dilation = (1,) * ndim

        self.register_parameter("weight", w_init(w_shape, jnp.float32, w_rng_key))
        b_shape = [out_features]
        self.register_parameter("bias", b_init(b_shape, jnp.float32, b_rng_key))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert len(x.shape) == len(self.kernel_format)

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
        )

        if self.with_bias:
            x = x + self.bias

        return x

    def __repr__(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "data_format": self.data_format,
            "padding": self.padding,
            "stride": self.stride,
            "rate": self.rate,
            "with_bias": self.with_bias,
        }
        return super().__repr__(info)


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
        w_init: Optional[initializers.Initializer] = None,
        b_init: Optional[initializers.Initializer] = None,
        data_format: str = "NWC",
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
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
                for each spatial dimension. Defaults to ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
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
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
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
        w_init: Optional[initializers.Initializer] = None,
        b_init: Optional[initializers.Initializer] = None,
        data_format: str = "NHWC",
        *,
        name: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None,
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
                for each spatial dimension. Defaults to ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
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
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            name=name,
            rng_key=rng_key,
        )
