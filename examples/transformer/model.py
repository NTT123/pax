import math
from typing import Sequence

import jax
import jax.numpy as jnp
import pax
from pax.nets import Transformer


def positional_encoding(x):
    _, L, D = x.shape
    position = jnp.arange(0, L, dtype=x.dtype)[:, None]
    div_term = jnp.exp(jnp.arange(0, D, 2, dtype=x.dtype) * (-math.log(10_000.0) / D))
    x1 = jnp.sin(position * div_term[None, :])
    x2 = jnp.cos(position * div_term[None, :])
    x_pos = jnp.concatenate((x1, x2), axis=-1)
    return x + x_pos[None, :, :]


class LM(pax.Module):
    """A Transformer language model."""

    transformer: Transformer
    embed: pax.Module
    output: pax.Module

    vocab_size: int
    hidden_dim: int

    def __init__(
        self, vocab_size: int, hidden_dim: int, num_layers: int, dropout: float = 0.1
    ):
        """
        Arguments:
            vocab_size: int, size of the alphabet.
            hidden_dim: int, hidden dim.
            num_layers: int, num transformer blocks.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = pax.Embed(
            vocab_size,
            hidden_dim,
            w_init=jax.nn.initializers.variance_scaling(
                1.0, mode="fan_out", distribution="normal"
            ),
        )
        self.transformer = Transformer(
            hidden_dim, hidden_dim // 64, num_layers, dropout_rate=dropout
        )
        self.output = pax.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        x = positional_encoding(x)
        x = self.transformer(x)
        logits = self.output(x)
        return logits

    @pax.pure
    def inference(self, prompt: Sequence[int] = (), length=1024, train_seq_len=256):
        def step(inputs, _):
            logits = self(inputs)
            x = jnp.argmax(logits[:, -1], axis=-1)
            next_inputs = jnp.concatenate((inputs[:, 1:], x[:, None]), axis=-1)
            return next_inputs, x

        if len(prompt) > train_seq_len:
            inputs = prompt[-train_seq_len:]
        else:
            inputs = prompt
        pad_len = train_seq_len - len(inputs)
        padded_inputs = [0] * pad_len + inputs
        x = jnp.array([padded_inputs], dtype=jnp.int32)
        L = length - len(prompt)
        _, out = pax.scan(step, x, None, length=L, time_major=False)
        return prompt + out[0].tolist()
