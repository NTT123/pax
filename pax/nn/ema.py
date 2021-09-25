from typing import Any, Optional

import jax
import jax.numpy as jnp

from ..module import Module


class EMA(Module):
    """Exponential Moving Average (EMA) Module"""

    averages: Any
    decay_rate: float
    debias: Optional[jnp.ndarray] = None

    def __init__(self, initial_value, decay_rate: float, debias: bool = False):
        """Create a new EMA module.

        Arguments:
            initial_value: the initial value.
            decay_rate: the decay rate.
            debias: ignore the initial value to avoid biased estimates.
        """

        super().__init__()
        self.register_state_subtree("averages", initial_value)
        self.decay_rate = decay_rate
        if debias:
            self.register_state("debias", jnp.array(False))

    def __call__(self, xs):
        """Return the ema of `xs`. Also, update internal states."""
        if self.debias is not None:
            self.averages = jax.tree_map(
                lambda a, x: jnp.where(self.debias, a, x), self.averages, xs
            )

            self.debias = jnp.logical_or(self.debias, True)

        self.averages = jax.tree_map(
            lambda a, x: a * self.decay_rate + x * (1 - self.decay_rate),
            self.averages,
            xs,
        )

        return self.averages
