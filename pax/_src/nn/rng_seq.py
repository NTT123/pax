"""RngSeq module."""

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..core import StateModule, rng


class RngSeq(StateModule):
    """A module which generates an infinite sequence of rng keys."""

    _rng_key: rng.KeyArray

    def __init__(
        self, seed: Optional[int] = None, rng_key: Optional[rng.KeyArray] = None
    ):
        """Initialize a random key sequence.

        **Note**: ``rng_key`` has a higher priority than ``seed``.

        Arguments:
            seed: an integer seed.
            rng_key: a jax random key.
        """
        super().__init__()
        if rng_key is not None:
            rng_key_ = rng_key
        elif seed is not None:
            rng_key_ = jax.random.PRNGKey(seed)
        else:
            rng_key_ = rng.next_rng_key()

        if isinstance(rng_key_, (np.ndarray, jnp.ndarray)):
            self._rng_key = rng_key_
        else:
            raise ValueError("Impossible")

    def next_rng_key(
        self, num_keys: int = 1
    ) -> Union[rng.KeyArray, Sequence[rng.KeyArray]]:
        """Return the next random key of the sequence.

        **Note**:

        * Return a key if ``num_keys`` is ``1``,
        * Return a list of keys if ``num_keys`` is greater than ``1``.
        * This is not a deterministic sequence if values of ``num_keys`` are mixed randomly.

        Arguments:
            num_keys: return more than one key.
        """
        self._rng_key, *rng_keys = jax.random.split(self._rng_key, num_keys + 1)
        return rng_keys[0] if num_keys == 1 else rng_keys
