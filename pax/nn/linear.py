import haiku as hk
import jax.numpy as jnp
import numpy as np

from .. import initializers
from ..module import Module
from ..rng import next_rng_key


class Linear(Module):
    """A linear transformation is applied over the last dimension of the input."""

    W: jnp.ndarray
    b: jnp.ndarray

    # props
    in_dim: int
    out_dim: int
    with_bias: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        with_bias: bool = True,
        w_init=initializers.variance_scaling(),
        b_init=jnp.zeros,
        *,
        name=None,
        rng_key: jnp.ndarray = None,
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
        self.weight = None
        self.bias = None
        self.with_bias = with_bias
        self.f = hk.without_apply_rng(
            hk.transform(
                lambda x: hk.Linear(out_dim, with_bias, w_init=w_init, b_init=b_init)(x)
            )
        )
        rng_key = next_rng_key() if rng_key is None else rng_key
        params = self.f.init(rng_key, np.empty((1, self.in_dim), dtype=np.float32))[
            "linear"
        ]
        self.register_parameter("weight", params["w"])
        if self.with_bias:
            self.register_parameter("bias", params["b"])

    def __call__(self, x: np.ndarray) -> jnp.ndarray:
        """Applies a linear transformation to the inputs along the last dimension.

        Arguments:
            x: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        return self.f.apply({"linear": {"w": self.weight, "b": self.bias}}, x)

    def __repr__(self):
        name = f"({self.name}) " if self.name is not None else ""
        cls_name = self.__class__.__name__
        return f"{name}{cls_name}[in_dim={self.in_dim}, out_dim={self.out_dim}, with_bias={self.with_bias}]"
