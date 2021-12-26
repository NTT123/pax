"""Recurrent modules."""

from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from ..core import Module
from ..core.rng import KeyArray, next_rng_key
from .linear import Linear


class LSTMState(NamedTuple):
    """LSTMState."""

    hidden: jnp.ndarray
    cell: jnp.ndarray


class GRUState(NamedTuple):
    """GRUState."""

    hidden: jnp.ndarray


class VanillaRNNState(NamedTuple):
    """VanillaRNNState."""

    hidden: jnp.ndarray


class RNN(Module):
    """Base class for all recurrent modules."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def initial_state(self, batch_size):
        raise NotImplementedError()


class VanillaRNN(RNN):
    """Basic recurrent neural network."""

    input_dim: int
    hidden_dim: int
    fc: Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        rng_key: KeyArray = None,
        name: Optional[str] = None
    ):
        """Create a vanilla RNN module.

        Arguments:
            input_dim: input dimension.
            hidden_dim: hidden dimension.
            rng_key: random key.
            name: module name.
        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = Linear(
            input_dim + hidden_dim,
            hidden_dim,
            rng_key=rng_key,
            name="vanilla_rnn_fc",
        )

    def __call__(
        self, state: VanillaRNNState, x: jnp.ndarray
    ) -> Tuple[VanillaRNNState, jnp.ndarray]:
        """A single rnn step."""
        xh = jnp.concatenate((x, state.hidden), axis=-1)
        hidden = jnp.tanh(self.fc(xh))
        return VanillaRNNState(hidden), hidden

    def __repr__(self):
        info = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim}
        return self._repr(info)

    def initial_state(self, batch_size) -> VanillaRNNState:
        shape = (batch_size, self.hidden_dim)
        hidden = jnp.zeros(shape=shape, dtype=jnp.float32)
        return VanillaRNNState(hidden=hidden)


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
        w_init: Optional[Callable] = None,
        forget_gate_bias: float = 0.0,
        *,
        rng_key: KeyArray = None,
        name: Optional[str] = None
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
        self,
        state: LSTMState,
        x: jnp.ndarray,
    ) -> Tuple[LSTMState, jnp.ndarray]:
        """Do a single lstm step.


        Arguments:
            state: The current LSTM state.
            x: The input.
        """
        xh = jnp.concatenate((x, state.hidden), axis=-1)
        gated = self.fc(xh)
        i, g, f, o = jnp.split(gated, 4, axis=-1)
        f = jax.nn.sigmoid(f + self.forget_gate_bias)
        c = f * state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return LSTMState(h, c), h

    def __repr__(self):
        info = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim}
        return self._repr(info)

    def initial_state(self, batch_size) -> LSTMState:
        shape = (batch_size, self.hidden_dim)
        hidden = jnp.zeros(shape=shape, dtype=jnp.float32)
        cell = jnp.zeros(shape=shape, dtype=jnp.float32)
        return LSTMState(hidden=hidden, cell=cell)


class GRU(RNN):
    """This class implements the "fully gated unit" GRU.

    Reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit
    """

    input_dim: int
    hidden_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        rng_key: Optional[KeyArray] = None,
        name: Optional[str] = None
    ):
        """Create a GRU module.

        Arguments:
            input_dim: the input size.
            hidden_dim: the number of GRU cells.
        """
        super().__init__(name=name)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if rng_key is None:
            rng_key = next_rng_key()
        rng_key_1, rng_key_2 = jax.random.split(rng_key, 2)
        self.xh_zr_fc = Linear(
            (input_dim + hidden_dim), hidden_dim * 2, name="xh_to_zr", rng_key=rng_key_1
        )

        self.xh_h_fc = Linear(
            (input_dim + hidden_dim), hidden_dim, name="xh_to_h", rng_key=rng_key_2
        )

    def initial_state(self, batch_size: int) -> GRUState:
        """Create an all zeros initial state."""
        return GRUState(jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32))

    def __call__(self, state: GRUState, x) -> Tuple[GRUState, jnp.ndarray]:
        """Do a single gru step.

        Arguments:
            state: The current GRU state.
            x: The input.
        """
        hidden = state.hidden
        xh = jnp.concatenate((x, hidden), axis=-1)
        zr = jax.nn.sigmoid(self.xh_zr_fc(xh))
        z, r = jnp.split(zr, 2, axis=-1)

        xrh = jnp.concatenate((x, r * hidden), axis=-1)
        h_hat = jnp.tanh(self.xh_h_fc(xrh))
        h = (1 - z) * hidden + z * h_hat
        return GRUState(h), h

    def __repr__(self):
        info = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim}
        return self._repr(info)
