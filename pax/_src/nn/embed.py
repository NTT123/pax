"""Embed module."""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ..core import ParameterModule
from ..core.rng import KeyArray, next_rng_key


class Embed(ParameterModule):
    """Embed module maps integer values to real vectors.
    The embedded vectors are trainable.
    """

    weight: jnp.ndarray
    vocab_size: int
    embed_dim: int

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        w_init: Optional[Callable] = None,
        *,
        rng_key: Optional[KeyArray] = None,
        name: Optional[str] = None
    ):
        """
        An embed module.

        Arguments:
            vocab_size: the number of embedded vectors.
            embed_dim: the size of embedded vectors.
            w_init: weight initializer. Default: `truncated_normal`.
            name: module name.
        """

        super().__init__(name=name)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        shape = [vocab_size, embed_dim]

        if w_init is None:
            w_init = jax.nn.initializers.normal()

        if rng_key is None:
            rng_key = next_rng_key()

        self.weight = w_init(rng_key, shape)

    def __call__(self, x: jnp.ndarray):
        """Return embedded vectors indexed by ``x``."""
        return self.weight[(x,)]

    def __repr__(self):
        info = {"vocab_size": self.vocab_size, "embed_dim": self.embed_dim}
        return self._repr(info)
