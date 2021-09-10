"""Useful functions."""

import inspect
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp

from . import rng
from .module import Module

T = TypeVar("T", bound="Module")

LossFnOutput = Tuple[jnp.ndarray, Tuple[Any, T]]
LossFn = Callable[[T, T, Any], LossFnOutput]
UpdateFn = Callable[[T, Module, Any], Tuple[Any, T, Module]]


def build_update_fn(loss_fn: LossFn) -> UpdateFn:
    """Build a simple update function.

    This function can be very useful. However, you have to follow its requirements *exactly*.
    This is to make sure you know exactly what you are doing.

    * The input ``loss_fn`` function has three parameters with names: ``params``, ``model``, ``inputs``.
    * ``loss_fn``'s output be annotated with type ``LossFnOutput``.

    Example:

    >>> def mse_loss(params, model, inputs) -> pax.utils.LossFnOutput:
    ...     model = model.update(params)
    ...     x, y = inputs
    ...     y_hat = model(x)
    ...     loss = jnp.mean(jnp.square(y - y_hat))
    ...     return loss, (loss, model)

    The returned ``update_fn`` function is:

    >>> def _update_fn(model: T, optimizer: Module, inputs: Any):
    ...     grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(
    ...         model.parameters(), model, inputs
    ...     )
    ...     model = model.update(
    ...         optimizer.step(grads, model.parameters()),
    ...     )
    ...     return loss, model, optimizer
    """

    sig = inspect.signature(loss_fn)
    parameters = sig.parameters
    if (
        list(parameters.keys()) != ["params", "model", "inputs"]
        or sig.return_annotation != LossFnOutput
    ):
        raise ValueError(
            """Expecting a loss function with an _exact_ signature:  
        ``(params, model, inputs) -> LossFnOutput``
        """
        )

    def _update_fn(model: T, optimizer: Module, inputs: Any):
        """An update function.

        Note that: ``model`` and ``optimizer`` have internal states.
        We have to return them in the output as jax transformations (e.g., ``jax.grad`` and ``jax.jit``) requires pure functions.


        Arguments:
            model: a callable pax.Module
            optimizer: an optimizer
            inputs: input batch.

        Returns:
            loss: the loss value
            model: updated model
            optimizer: updated optimizer
        """
        grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(
            model.parameters(), model, inputs
        )
        model = model.update(
            optimizer.step(grads, model.parameters()),
        )
        return loss, model, optimizer

    return _update_fn


def dropout(rng_key: jnp.ndarray, dropout_rate: float, x: jnp.ndarray) -> jnp.ndarray:
    """dropout input `x` randomly.

    Scaling the input by ``1 / (1-dropout_rate)`` makes ``E[output] = input``.
    """
    assert 0 <= dropout_rate < 1.0

    if dropout_rate == 0.0:
        return x
    else:
        mask = jax.random.bernoulli(rng_key, dropout_rate, shape=x.shape)
        x = jnp.where(mask, 0.0, x / (1.0 - dropout_rate))
        return x


class Lambda(Module):
    """A pure functional module.

    Note: We put ``Lambda`` module definition here so both ``haiku`` and ``nn`` modules can use it.
    """

    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.f}]"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.f == jax.nn.relu:
            name = "relu"
        else:
            name = f"{self.f}"
        output = f"x => {name}(x)"
        return [output] if return_list else output


class RngSeq(Module):
    """A module which generates an infinite sequence of rng keys."""

    _rng_key: jnp.ndarray

    def __init__(self, seed: int = None, rng_key: jnp.ndarray = None):
        """Initialize a random key sequence.

        **Note**: ``rng_key`` has higher priority than ``seed``.

        Arguments:
            seed: an integer seed.
            rng_key: a jax random key.
        """
        super().__init__()
        if rng_key is not None:
            rng_key = rng_key
        elif seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            rng_key = rng.next_rng_key()

        self.register_state("_rng_key", rng_key)

    def next_rng_key(
        self, num_keys: int = 1
    ) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
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
        if self.training:
            self._rng_key = _rng_key
        if num_keys == 1:
            return rng_keys[0]
        else:
            return rng_keys


class EMA(Module):
    """Exponential Moving Average (EMA) Module"""

    averages: Any
    decay_rate: float
    debias: Optional[jnp.ndarray] = None

    def __init__(self, initial_value, decay_rate: float, debias: bool = False):
        """Create a new EMA module.

        Arguments:
            initial_value: the initial value.
            decay_rate: the decay rate.
            debias: ignore the initial value to avoid biased estimates.
        """

        super().__init__()
        self.register_state_subtree("averages", initial_value)
        self.decay_rate = decay_rate
        if debias:
            self.register_state("debias", jnp.array(False))

    def __call__(self, xs):
        if self.debias is not None:
            self.averages = jax.tree_map(
                lambda a, x: jnp.where(self.debias, a, x), self.averages, xs
            )

            self.debias = jnp.logical_or(self.debias, True)

        self.averages = jax.tree_map(
            lambda a, x: a * self.decay_rate + x * (1 - self.decay_rate),
            self.averages,
            xs,
        )

        return self.averages
