"""Useful functions."""

import inspect
from typing import Any, Callable, Tuple, TypeVar
from unittest import TestCase

import jax
import jax.numpy as jnp

from pax.transforms import apply_updates, grad_with_aux, transform_gradient

from .module import Module, PaxFieldKind
from .rng import KeyArray

T = TypeVar("T", bound="Module")

LossFnOutput = Tuple[jnp.ndarray, Tuple[Any, T]]
LossFn = Callable[[T, T, Any], LossFnOutput]


GradientTransformation = "GradientTransformation"
UpdateFn = Callable[
    [Tuple[T, GradientTransformation], Any],
    Tuple[Tuple[T, GradientTransformation], Any],
]


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

    >>> def _update_fn(model_and_optimizer: Tuple[Module, GradientTransformation], inputs: Any):
    ...     model, optimizer = model_and_optimizer
    ...     grads, (loss, model) = pax.grad(loss_fn, has_aux=True)(
    ...         pax.select_parameter(model), model, inputs
    ...     )
    ...     model = model.update(
    ...         optimizer.step(grads, pax.select_parameter(model)),
    ...     )
    ...     return (model, optimizer), loss
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

    def _update_fn(
        model_and_optimizer: Tuple[Module, GradientTransformation], inputs: Any
    ) -> Tuple[Tuple[Module, GradientTransformation], Any]:
        """An update function.

        Note that: ``model`` and ``optimizer`` have internal states.
        We have to return them in the output as jax transformations (e.g., ``jax.grad`` and ``jax.jit``) requires pure functions.


        Arguments:
            model_and_optimizer: (a callable pax.Module, an optimizer),
            inputs: input batch.

        Returns:
            model_and_optimizer: updated (model, optimizer),
            aux: the aux info.
        """
        model, optimizer = model_and_optimizer
        from .transforms import select_parameter

        params = select_parameter(model)
        grads, (aux, model) = grad_with_aux(model, fn=loss_fn, inputs=inputs)
        assertStructureEqual(grads, select_parameter(model))
        updates, optimizer = transform_gradient(
            grads, params=params, optimizer=optimizer
        )
        params = apply_updates(params, updates=updates)
        model = model.update(params)
        return (model, optimizer), aux

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


def assertStructureEqual(self: T, other: T):
    """Assert that the two modules are structurally the same.

    Print out the difference.
    """
    if jax.tree_structure(self) == jax.tree_structure(other):
        return True

    def check(a, b):
        if isinstance(a, Module) and isinstance(b, Module):
            assertStructureEqual(a, b)

    tc = TestCase()
    tc.maxDiff = None

    def filter_out_module(d):
        return {
            k: ((v.shape, v.dtype) if isinstance(v, jnp.ndarray) else v)
            for (k, v) in d.items()
            if (k not in self._name_to_kind)
            or (
                self._name_to_kind[k]
                not in [PaxFieldKind.MODULE, PaxFieldKind.MODULE_SUBTREE]
            )
        }

    tc.assertDictEqual(filter_out_module(vars(self)), filter_out_module(vars(other)))

    jax.tree_map(
        check,
        self,
        other,
        is_leaf=lambda x: isinstance(x, Module) and x is not self and x is not other,
    )
