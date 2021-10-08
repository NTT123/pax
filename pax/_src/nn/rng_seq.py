from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .. import rng
from ..module import Module


class RngSeq(Module):
    """A module which generates an infinite sequence of rng keys."""

    _rng_key: rng.KeyArray

    def __init__(
        self, seed: Optional[int] = None, rng_key: Optional[rng.KeyArray] = None
    ):
        """Initialize a random key sequence.

        **Note**: ``rng_key`` has higher priority than ``seed``.

        Arguments:
            seed: an integer seed.
            rng_key: a jax random key.
        """
        super().__init__()
        if rng_key is not None:
            _rng_key = rng_key
        elif seed is not None:
            _rng_key = jax.random.PRNGKey(seed)
        else:
            _rng_key = rng.next_rng_key()

        if isinstance(_rng_key, (np.ndarray, jnp.ndarray)):
            self.register_state("_rng_key", _rng_key)
        else:
            raise ValueError("Impossible")

    def next_rng_key(
        self, num_keys: int = 1
    ) -> Union[rng.KeyArray, Sequence[rng.KeyArray]]:
        """Return the next random key of the sequence.

        **Note**:

            * Return a key if ``num_keys`` is ``1``,
            * Return a list of keys if ``num_keys`` is greater than ``1``.
            * This is not a deterministic sequence if values of ``num_keys`` is mixed randomly.

        Arguments:
            num_keys: return more than one key.
        """
        _rng_key, *rng_keys = jax.random.split(self._rng_key, num_keys + 1)

        # only update internal state in `train` mode.
        if self.is_training():
            self._rng_key = _rng_key
        if num_keys == 1:
            return rng_keys[0]
        else:
            return rng_keys
