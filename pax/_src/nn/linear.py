"""Linear module."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .. import initializers
from ..core import Module
from ..core.rng import KeyArray, next_rng_key


class Linear(Module):
    """A linear transformation is applied over the last dimension of the input."""

    weight: jnp.ndarray
    bias: jnp.ndarray

    in_dim: int
    out_dim: int
    with_bias: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        with_bias: bool = True,
        w_init=None,
        b_init=None,
        *,
        name: Optional[str] = None,
        rng_key: KeyArray = None,
    ):
        """
        Arguments:
            in_dim: the number of input features.
            out_dim: the number of output features.
            with_bias: whether to add a bias to the output (default: True).
            w_init: initializer function for the weight matrix.
            b_init: initializer function for the bias.
            rng_key: the key to generate initial parameters.
        """
        super().__init__(name=name)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.with_bias = with_bias

        rng_key = next_rng_key() if rng_key is None else rng_key
        if w_init is None:
            w_init = initializers.truncated_normal(stddev=1.0 / np.sqrt(self.in_dim))
        b_init = initializers.zeros
        rng_key_w, rng_key_b = jax.random.split(rng_key)
        self.register_parameter(
            "weight", w_init((in_dim, out_dim), jnp.float32, rng_key_w)
        )
        if self.with_bias:
            self.register_parameter("bias", b_init((out_dim,), jnp.float32, rng_key_b))

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Arguments:
            x: The nd-array to be transformed.
        """
        assert len(x.shape) >= 2, "expecting an input of shape `N...C`"
        x = jnp.dot(x, self.weight)
        if self.with_bias:
            x = x + self.bias
        return x

    def __repr__(self):
        info = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "with_bias": self.with_bias,
        }
        return super()._repr(info)
