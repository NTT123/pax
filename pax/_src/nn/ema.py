"""EMA module."""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from ..core import StateModule


def _has_integer_leaves(x):
    """check if there is any interger/bool leaves"""
    leaves = jax.tree_leaves(x)
    return not all(jnp.issubdtype(leaf, jnp.floating) for leaf in leaves)


class EMA(StateModule):
    """Exponential Moving Average (EMA) Module"""

    averages: Any
    decay_rate: float
    debias: Optional[jnp.ndarray]
    allow_int: bool

    def __init__(
        self,
        initial_value,
        decay_rate: float,
        debias: bool = False,
        allow_int: bool = False,
    ):
        """Create a new EMA module.

        If allow_int=True, integer leaves are updated to
        the newest values instead of averaging.

        Arguments:
            initial_value: the initial value.
            decay_rate: the decay rate.
            debias: ignore the initial value to avoid biased estimates.
            allow_int: allow integer values.
        """
        if not allow_int:
            if _has_integer_leaves(initial_value):
                raise ValueError(
                    "There are integer arrays in the initial value.\n"
                    "Use `allow_int=True` to allow this."
                )

        super().__init__()
        self.averages = initial_value
        self.decay_rate = decay_rate
        self.allow_int = allow_int
        if debias:
            # avoid integer ndarray for `jax.grad` convenience,
            # e.g., no need to pass `allow_int=True` to `jax.grad`.
            self.debias = jnp.array(0.0)
        else:
            self.debias = None

    def __call__(self, xs):
        """Return the ema of `xs`. Also, update internal states."""
        if not self.allow_int:
            if _has_integer_leaves(xs):
                raise ValueError(
                    "There are integer arrays in the new value.\n"
                    "Use `allow_int=True` to allow this."
                )

        if self.training:
            if self.debias is not None:
                cond = self.debias > 0
                debias_func = lambda a, x: jnp.where(cond, a, x)
                self.debias = jnp.array(1.0)
            else:
                debias_func = lambda a, _: a

            def update_fn(a, x):
                if jnp.issubdtype(a, jnp.floating):
                    a = debias_func(a, x)
                    return a * self.decay_rate + x * (1 - self.decay_rate)
                else:
                    return x

            self.averages = jax.tree_map(update_fn, self.averages, xs)

        return self.averages
