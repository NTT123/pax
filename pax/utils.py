"""Useful functions."""

import inspect
from typing import Any, Callable, Tuple, TypeVar

import jax
import jax.numpy as jnp

from .module import Module
from .pax_transforms import grad
from .rng import KeyArray

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
    ...     grads, (loss, model) = pax.grad(loss_fn, has_aux=True)(
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

    from opax import GradientTransformation

    def _update_fn(model: T, optimizer: GradientTransformation, inputs: Any):
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
        grads, (loss, model) = grad(loss_fn, has_aux=True)(
            model.parameters(), model, inputs
        )
        grads.assertStructureEqual(model.parameters())
        model = model.update(
            optimizer.step(grads, model.parameters()),
        )
        return loss, model, optimizer

    return _update_fn


def dropout(rng_key: KeyArray, dropout_rate: float, x: jnp.ndarray) -> jnp.ndarray:
    """Dropout input `x` randomly.

    Scaling the input by ``1 / (1-dropout_rate)`` makes ``E[output] = input``.
    """
    assert 0 <= dropout_rate < 1.0

    if dropout_rate == 0.0:
        return x
    else:
        mask = jax.random.bernoulli(rng_key, dropout_rate, shape=x.shape)
        x = jnp.where(mask, 0.0, x / (1.0 - dropout_rate))
        return x


def scan(fn, init, xs, length=None, unroll: int = 1, time_major=True):
    """``jax.lax.scan`` with an additional ``time_major=False`` mode.


    The semantics of ``scan`` are given roughly by this Python implementation::

      def scan(f, init, xs, length=None):
          if xs is None:
              xs = [None] * length
          carry = init
          ys = []
          for x in xs:
              carry, y = f(carry, x)
              ys.append(y)
          return carry, np.stack(ys)

    """
    if time_major:
        # data format: TN...
        return jax.lax.scan(fn, init, xs, length=length, unroll=unroll)
    else:
        # data format: NT...
        if xs is not None:
            xs = jnp.swapaxes(xs, 0, 1)  # swap batch and time axes
        state, output = jax.lax.scan(fn, init, xs, length=length, unroll=unroll)
        output = jnp.swapaxes(output, 0, 1)  # restore to NT...
        return state, output
