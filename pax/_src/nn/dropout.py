"""Dropout module."""

from typing import Optional

import jax
import jax.numpy as jnp

from ..core import StateModule
from ..core.rng import KeyArray, next_rng_key


def dropout(rng_key: KeyArray, dropout_rate: float, x: jnp.ndarray) -> jnp.ndarray:
    """Dropout input `x` randomly.

    Scaling the input by ``1 / (1-dropout_rate)`` makes ``E[output] = input``.
    """
    assert 0 <= dropout_rate < 1.0

    if dropout_rate == 0.0:
        return x
    else:
        mask = jax.random.bernoulli(rng_key, dropout_rate, shape=x.shape)
        x = jnp.where(mask, 0.0, x / (1.0 - dropout_rate))
        return x


class Dropout(StateModule):
    """A Dropout Module.

    Dropout module stores an internal state ``rng_key``.
    It refreshes ``rng_key`` whenever a forward pass is executed.
    """

    rng_key: KeyArray
    dropout_rate: float

    def __init__(self, dropout_rate: float, *, name: Optional[str] = None):
        """Create a dropout module.

        Arguments:
            dropout_rate: the probability of dropping an element.
            name: the module name.
        """
        super().__init__(name=name)
        assert 0 <= dropout_rate < 1.0

        self.dropout_rate = dropout_rate
        self.rng_key = next_rng_key()

    def __call__(self, x):
        """Dropout `x` randomly.

        Return the input `x` if in `eval` mode or `dropout_rate=0`.
        """

        if self.training and self.dropout_rate > 0:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
            return dropout(rng_key, self.dropout_rate, x)
        else:
            return x

    def __repr__(self):
        return super()._repr({"dropout_rate": self.dropout_rate})
