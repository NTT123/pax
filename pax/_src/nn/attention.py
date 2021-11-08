"""Transformer self-attention module."""

# Source: https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/attention.py
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..core import Module
from .linear import Linear


class MultiHeadAttention(Module):
    """Multi-headed attention mechanism.
    As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
    """

    num_heads: int
    key_size: int
    value_size: int
    model_size: int

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init_scale: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = key_size
        self.model_size = key_size * num_heads
        w_init = jax.nn.initializers.variance_scaling(
            w_init_scale, mode="fan_in", distribution="normal"
        )
        self.query_projection = Linear(
            self.model_size, self.model_size, w_init=w_init, name="qry_proj"
        )
        self.key_projection = Linear(
            self.model_size, self.model_size, w_init=w_init, name="key_proj"
        )
        self.value_projection = Linear(
            self.model_size, self.model_size, w_init=w_init, name="val_proj"
        )
        self.output_projection = Linear(
            self.model_size, self.model_size, w_init=w_init, name="out_proj"
        )

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""

        query_heads = self.query_projection(query)
        key_heads = self.key_projection(key)
        value_heads = self.value_projection(value)
        (query_heads, key_heads, value_heads) = jax.tree_map(
            lambda x, y: x.reshape(*y.shape[:-1], self.num_heads, self.key_size),
            (query_heads, key_heads, value_heads),
            (query, key, value),
        )

        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.key_size).astype(key.dtype)
        attn_logits = attn_logits / sqrt_key_size
        if mask is not None:
            # assert mask.shape == attn_logits.shape
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits)
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return self.output_projection(attn_vec)

    def __repr__(self, info=None) -> str:
        info = {"num_heads": self.num_heads, "key_size": self.key_size}
        return self._repr(info)
