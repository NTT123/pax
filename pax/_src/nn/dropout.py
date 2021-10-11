from typing import Optional

import jax

from ..core import Module
from ..rng import KeyArray, next_rng_key
from ..utils import dropout


class Dropout(Module):
    """A Dropout Module.

    Dropout module stores an internal state ``rng_key``. It refreshes ``rng_key`` whenever a forward pass is executed.
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
        self.register_state("rng_key", next_rng_key())

    def __call__(self, x):
        """Dropout `x` randomly.

        Return the input `x` if in `eval` mode or `dropout_rate=0`.
        """

        if self.training and self.dropout_rate > 0:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
            return dropout(rng_key, self.dropout_rate, x)
        else:
            return x
