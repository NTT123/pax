from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pax
from pax.nn import MultiHeadAttention


class CausalSelfAttention(MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        value: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)


class DenseBlock(pax.Module):
    """A 2-layer MLP which widens then narrows the input."""

    def __init__(self, in_dim: int, init_scale: float, widening_factor: int = 4):
        super().__init__()
        self._init_scale = init_scale
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        self._widening_factor = widening_factor
        self.fc1 = pax.nn.Linear(in_dim, in_dim * widening_factor, w_init=initializer)
        self.fc2 = pax.nn.Linear(in_dim * widening_factor, in_dim, w_init=initializer)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        return self.fc2(x)


class Transformer(pax.Module):
    """A transformer stack."""

    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout_rate: float):
        super().__init__()
        assert dim % num_heads == 0
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

        self.register_state("rng_key", pax.next_rng_key())

        init_scale = 2.0 / self._num_layers
        layers = []
        for _ in range(num_layers):
            layers.append(
                {
                    "attention": CausalSelfAttention(
                        num_heads=self._num_heads,
                        key_size=dim // num_heads,
                        w_init_scale=init_scale,
                    ),
                    "attn_layer_norm": pax.nn.LayerNorm(dim, -1, True, True),
                    "dense_layer_norm": pax.nn.LayerNorm(dim, -1, True, True),
                    "dense_block": DenseBlock(dim, init_scale),
                }
            )
        self.register_module_subtree("layers", layers)
        self.layer_norm_output = pax.nn.LayerNorm(dim, -1, True, True)

    def __call__(
        self, h: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Connects the transformer.
        Args:
          h: Inputs, [B, T, H].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.
        Returns:
          Array of shape [B, T, H].
        """

        dropout_rate = self._dropout_rate if self.training else 0.0
        if mask is not None:
            mask = mask[:, None, None, :]

        # Note: names chosen to approximately match those used in the GPT-2 code;
        # see https://github.com/openai/gpt-2/blob/master/src/model.py.
        rng_key, self.rng_key = jax.random.split(self.rng_key)
        rngs = jax.random.split(rng_key, self._num_layers * 2)
        for i in range(self._num_layers):
            h_norm = self.layers[i]["attn_layer_norm"](h)
            h_attn = self.layers[i]["attention"](h_norm, mask=mask)
            h_attn = pax.haiku.dropout(rngs[i * 2 + 0], dropout_rate, h_attn)
            h = h + h_attn
            h_norm = self.layers[i]["dense_layer_norm"](h)
            h_dense = self.layers[i]["dense_block"](h_norm)
            h_dense = pax.haiku.dropout(rngs[i * 2 + 1], dropout_rate, h_dense)
            h = h + h_dense
        h = self.layer_norm_output(h)

        return h
