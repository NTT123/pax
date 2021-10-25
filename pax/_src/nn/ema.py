"""EMA module."""

from typing import Any, Optional, TypeVar

import jax
import jax.numpy as jnp

from ..core import Module

T = TypeVar("T")


class EMA(Module):
    """Exponential Moving Average (EMA) Module"""

    averages: Any
    decay_rate: float
    debias: Optional[jnp.ndarray]

    def __init__(self, initial_value, decay_rate: float, debias: bool = False):
        """Create a new EMA module.

        Arguments:
            initial_value: the initial value.
            decay_rate: the decay rate.
            debias: ignore the initial value to avoid biased estimates.
        """

        super().__init__()
        self.register_states("averages", initial_value)
        self.decay_rate = decay_rate
        if debias:
            # avoid integer ndarray for `jax.grad` convenience,
            # e.g., no need to pass `allow_int=True` to `jax.grad`.
            self.register_states("debias", jnp.array(0.0))
        else:
            self.debias = None

    def __call__(self, xs: T) -> T:
        """Return the ema of `xs`. Also, update internal states."""

        if self.training:
            if self.debias is not None:
                cond = self.debias > 0
                self.averages = jax.tree_map(
                    lambda a, x: jnp.where(cond, a, x), self.averages, xs
                )

                self.debias = jnp.array(1.0)

            self.averages = jax.tree_map(
                lambda a, x: a * self.decay_rate + x * (1 - self.decay_rate),
                self.averages,
                xs,
            )

        return self.averages
