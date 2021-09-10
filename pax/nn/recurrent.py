"""Recurrent Modules."""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from ..initializers import Initializer
from ..module import Module
from .linear import Linear


class RNN(Module):
    """Base class for all recurrent modules."""

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def initial_state(self, batch_size):
        raise NotImplementedError()


class LSTMState(NamedTuple):
    hidden: jnp.ndarray
    cell: jnp.ndarray


class LSTM(RNN):
    """Long Short Term Memory (LSTM) RNN module."""

    input_dim: int
    hidden_dim: int

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        w_init: Initializer = None,
        forget_gate_bias: float = 0.0,
        *,
        rng_key: jnp.ndarray = None,
        name: str = None
    ):
        """Create a LSTM module.

        Arguments:
            input_dim: The input dimension.
            hidden_dim: The number of LSTM cells.
            w_init: weight initializer.
            forget_gate_bias: Prefer forget. Default `0`.
            rng_key: random key.
            name: module name.
        """

        super().__init__(name=name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_gate_bias = forget_gate_bias

        self.fc = Linear(
            (input_dim + hidden_dim),
            4 * hidden_dim,
            rng_key=rng_key,
            name="lstm_fc",
            w_init=w_init,
        )

    def __call__(
        self, x: jnp.ndarray, state: LSTMState
    ) -> Tuple[jnp.ndarray, LSTMState]:
        xh = jnp.concatenate((x, state.hidden), axis=-1)
        gated = self.fc(xh)
        i, g, f, o = jnp.split(gated, 4, axis=-1)
        f = jax.nn.sigmoid(f + self.forget_gate_bias)
        c = f * state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    def __repr__(self):
        info = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim}
        return super().__repr__(info)

    def initial_state(self, batch_size):
        shape = (batch_size, self.hidden_dim)
        hidden = jnp.zeros(shape=shape, dtype=jnp.float32)
        cell = jnp.zeros(shape=shape, dtype=jnp.float32)
        return LSTMState(hidden=hidden, cell=cell)
