"""Convert Haiku module to pax.Module"""
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from haiku import LSTMState, dynamic_unroll
from pax.module import Module

HaikuState = Dict[str, Dict[str, jnp.ndarray]]
HaikuParam = Dict[str, Dict[str, jnp.ndarray]]


def from_haiku(cls, use_rng: bool = False, pass_is_training: bool = False, **kwargs):
    """Build a pax.Module class from haiku cls.

    Arguments:
        cls: dm-haiku class. For example, hk.BatchNorm.
        use_rng: generate a new rng key for each call.
        pass_is_training: pass `is_training` as an argument to module. `hk.BatchNorm` needs this.

    Returns:
        A pax.Module class.
    """
    fwd = lambda *u, **v: cls(**kwargs)(*u, **v)
    hk_fwd = hk.transform_with_state(fwd)

    class HaikuModule(Module):
        params: HaikuParam
        state: HaikuState
        rng_key: jnp.ndarray

        def __init__(self, *u, rng_key: Optional[jnp.ndarray] = None, **v) -> None:
            super().__init__()
            rng_key = jax.random.PRNGKey(42) if rng_key is None else rng_key
            if pass_is_training:
                v["is_training"] = self.training
            params, state = map(
                hk.data_structures.to_mutable_dict,
                hk_fwd.init(rng_key, *u, **v),
            )
            self.register_parameter_subtree("params", params)
            self.register_state_subtree("state", state)
            self.register_state_subtree("rng_key", rng_key)

        def __call__(self, *args, **kwargs):
            if use_rng:
                new_rng_key, rng_key = jax.random.split(self.rng_key, 2)
            else:
                rng_key = None
            if pass_is_training:
                kwargs["is_training"] = self.training
            out, state = hk_fwd.apply(self.params, self.state, rng_key, *args, **kwargs)
            if self.training:
                # only update state in training mode.
                if use_rng:
                    self.rng_key = new_rng_key
                self.state = hk.data_structures.to_mutable_dict(state)
            return out

    return HaikuModule


def batch_norm_2d(
    num_channels: int, axis: int = -1, decay_rate=0.99, cross_replica_axis=None
):
    """Return a BatchNorm module."""
    BatchNorm = from_haiku(
        hk.BatchNorm,
        pass_is_training=True,
        create_scale=True,
        create_offset=True,
        decay_rate=decay_rate,
        cross_replica_axis=cross_replica_axis,
    )
    shape = [1, 1, 1, 1]
    shape[axis] = num_channels
    x = np.empty((num_channels,), dtype=np.float32).reshape(shape)
    return BatchNorm(x)


def layer_norm(num_channels: int, axis: int = -1):
    LayerNorm = from_haiku(
        hk.LayerNorm, axis=axis, create_scale=True, create_offset=True
    )
    shape = [1, 1, 1, 1]
    shape[axis] = num_channels
    x = np.empty((num_channels,), dtype=np.float32).reshape(shape)
    return LayerNorm(x)


def linear(in_dim: int, out_dim: int, with_bias: bool = True):
    Linear = from_haiku(hk.Linear, output_size=out_dim, with_bias=with_bias)
    shape = (1, in_dim)
    x = np.empty(shape, dtype=np.float32)
    return Linear(x)


def lstm(hidden_dim: int):
    LSTM = from_haiku(hk.LSTM, hidden_size=hidden_dim)

    def initial_state(o, batch_size):
        h0 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        c0 = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        return LSTMState(h0, c0)

    LSTM.initial_state = initial_state
    x = np.empty((1, hidden_dim), dtype=np.float32)
    return LSTM(x, LSTM.initial_state(LSTM, 1))


def embed(vocab_size: int, embed_dim: int):
    Embed = from_haiku(hk.Embed, vocab_size=vocab_size, embed_dim=embed_dim)
    x = np.empty((1, 1), dtype=np.int32)
    return Embed(x)
