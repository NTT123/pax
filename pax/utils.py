"""Useful functions."""

import functools
import inspect
from typing import Any, Callable, Tuple, TypeVar, Union
from unittest import TestCase

import jax
import jax.numpy as jnp

from .module import Module, PaxFieldKind
from .rng import KeyArray
from .strict_mode import grad

GradientTransformation = "GradientTransformation"
T = TypeVar("T", bound=Module)
O = TypeVar("O", bound=GradientTransformation)
C = TypeVar("C")


LossFnOutput = Tuple[jnp.ndarray, Any]
LossFn = Callable[[T, Any], LossFnOutput]

UpdateFn_ = Callable[[T, O, Any], Tuple[T, O, Any]]
UpdateFnScan = Callable[[Tuple[T, O], Any], Tuple[Tuple[T, O], Any]]

UpdateFn = Union[UpdateFn_, UpdateFnScan]


@jax.tree_util.register_pytree_node_class
class EmptyNode(Tuple):
    """We use this class to mark deleted nodes.

    Note: this is inspired by treex's `Nothing` class.
    """

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _, __):
        return EmptyNode()


def grad_parameters(
    fun: Union[
        Callable[[T, Any], Tuple[jnp.ndarray, C]],
        Callable[[T, Any, Any], Tuple[jnp.ndarray, C]],
        Callable[[T, Any, Any, Any], Tuple[jnp.ndarray, C]],
        Callable[..., Tuple[jnp.ndarray, C]],
    ]
) -> Callable[..., Tuple[T, C]]:
    """Compute gradient with respect to trainable parameters of the first argument."""

    @functools.wraps(fun)
    def _fun(params: T, mod: T, *args, **kwargs):
        mod = mod.update_parameters(params.parameters())
        out = fun(mod, *args, **kwargs)
        return out

    _grad_fn = grad(_fun, has_aux=True, allow_int=False, io_check=True, copy=True)

    def grad_fn(mod: T, *args, **kwargs) -> Tuple[T, C]:
        if not isinstance(mod, Module):
            raise ValueError("Expecting a Pax's Module at the first argument.")

        out = _grad_fn(mod.parameters(), mod, *args, **kwargs)
        return out

    return grad_fn


def build_update_fn(loss_fn: LossFn, *, scan_mode: bool = False) -> UpdateFn:
    """Build a simple update function.

    This function can be very useful. However, you have to follow its requirements *exactly*.
    This is to make sure you know exactly what you are doing.

    * The input ``loss_fn`` function has three parameters with names: ``model``, ``inputs``.
    * ``loss_fn``'s output be annotated with type ``LossFnOutput``.

    Arguments:
        loss_fn: The loss function.
        scan_mode: If true, use `(model, optimizer)` as a single argument.

    Example:

    >>> def mse_loss(model, inputs) -> pax.LossFnOutput:
    ...     x, y = inputs
    ...     y_hat = model(x)
    ...     loss = jnp.mean(jnp.square(y - y_hat))
    ...     return loss, (loss, model)

    The returned ``update_fn`` function is:

    >>> def _update_fn(model: Module, optimizer: GradientTransformation, inputs: Any):
    ...     grads, (aux, model) = pax.grad(loss_fn, hax_aux=True, allow_int=True)(model, inputs)
    ...     assertStructureEqual(grads, model)
    ...     params = select_parameters(model)
    ...     updates, optimizer = transform_gradients(grads, optimizer, params=params)
    ...     params = apply_updates(params, updates=updates)
    ...     model = update_parameters(model, params=params)
    ...     return model, optimizer, aux
    """

    sig = inspect.signature(loss_fn)
    parameters = sig.parameters
    if (
        list(parameters.keys()) != ["model", "inputs"]
        or sig.return_annotation != LossFnOutput
    ):
        raise ValueError(
            """Expecting a loss function with an _exact_ signature:  
        ``(model, inputs) -> pax.LossFnOutput``
        """
        )

    def _update_fn(model: T, optimizer: O, inputs: Any) -> Tuple[T, O, Any]:
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

        from .strict_mode import grad
        from .transforms import (
            apply_updates,
            select_parameters,
            transform_gradients,
            update_parameters,
        )

        grads, (aux, model) = grad_parameters(loss_fn)(model, inputs)
        params = select_parameters(model)
        updates, optimizer = transform_gradients(grads, optimizer, params=params)
        params = apply_updates(params, updates=updates)
        model = update_parameters(model, params=params)
        return model, optimizer, aux

    def _update_fn_scan(
        model_and_optimizer: Tuple[T, O], inputs: Any
    ) -> Tuple[Tuple[T, O], Any]:
        model, optimizer = model_and_optimizer
        model, optimizer, aux = _update_fn(model, optimizer, inputs)
        return (model, optimizer), aux

    return _update_fn_scan if scan_mode else _update_fn


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
            or (self._name_to_kind[k] != PaxFieldKind.MODULE)
        }

    tc.assertDictEqual(filter_out_module(vars(self)), filter_out_module(vars(other)))

    jax.tree_map(
        check,
        self,
        other,
        is_leaf=lambda x: isinstance(x, Module) and x is not self and x is not other,
    )
