"""Useful functions."""

import functools
from typing import Any, Callable, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp

from .core import Module, select_parameters, update_parameters

GradientTransformation = Module
T = TypeVar("T", bound=Module)
O = TypeVar("O", bound=GradientTransformation)
C = TypeVar("C")


@functools.wraps(jax.grad)
def grad_parameters(
    fun: Union[
        Callable[[T, Any], Tuple[jnp.ndarray, C]],
        Callable[[T, Any, Any], Tuple[jnp.ndarray, C]],
        Callable[[T, Any, Any, Any], Tuple[jnp.ndarray, C]],
        Callable[..., Tuple[jnp.ndarray, C]],
    ],
    *args,
    **kwargs
) -> Callable[..., Tuple[T, C]]:
    """Compute gradient with respect to trainable parameters of the first argument.

    Example:

    >>> @pax.pure
    ... def loss_fn(model: pax.nn.Linear, x, y):
    ...     y_hat = model(x)
    ...     loss = jnp.mean(jnp.square(y - y_hat))
    ...     return loss, (loss, model)
    ...
    >>> grad_fn = pax.grad_parameters(loss_fn)
    >>> net = pax.nn.Linear(1, 1)
    >>> x = jnp.zeros((3, 1))
    >>> grads, (loss, net) = grad_fn(net, x, x)
    """

    def _fun(params: T, mod: T, *args, **kwargs):
        mod = mod.update_parameters(params.parameters())
        out = fun(mod, *args, **kwargs)
        return out

    grad_fn_ = jax.grad(_fun, *args, **kwargs)

    def grad_fn(mod: T, *args, **kwargs) -> Tuple[T, C]:
        if not isinstance(mod, Module):
            raise ValueError("Expecting a PAX's Module at the first argument.")

        out = grad_fn_(mod.parameters(), mod, *args, **kwargs)
        return out

    return grad_fn


def build_update_fn(loss_fn, *, scan_mode: bool = False):
    """Build a simple update function.

    *Note*: The output of ``loss_fn`` must be ``(loss, (aux, model))``.

    Arguments:
        loss_fn: The loss function.
        scan_mode: If true, use `(model, optimizer)` as a single argument.

    Example:

    >>> def mse_loss(model, x, y):
    ...     y_hat = model(x)
    ...     loss = jnp.mean(jnp.square(y - y_hat))
    ...     return loss, (loss, model)
    ...
    >>> update_fn = pax.utils.build_update_fn(mse_loss)
    >>> net = pax.nn.Linear(2, 2)
    >>> optimizer = opax.adam(1e-4)(net.parameters())
    >>> x = jnp.ones((32, 2))
    >>> y = jnp.zeros((32, 2))
    >>> net, optimizer, loss = update_fn(net, optimizer, x, y)
    """

    # pylint: disable=import-outside-toplevel
    from opax import apply_updates, transform_gradients

    def _update_fn(model: T, optimizer: O, *inputs, **kwinputs) -> Tuple[T, O, Any]:
        """An update function.

        Note that: ``model`` and ``optimizer`` have internal states.
        We have to return them in the output as jax transformations
        (e.g., ``jax.grad`` and ``jax.jit``) requires pure functions.


        Arguments:
            model_and_optimizer: (a callable pax.Module, an optimizer),
            inputs: input batch.

        Returns:
            model_and_optimizer: updated (model, optimizer),
            aux: the aux info.
        """

        assert isinstance(model, Module)
        assert isinstance(optimizer, Module)

        model_treedef = jax.tree_structure(model)
        grads, (aux, model) = grad_parameters(loss_fn, has_aux=True)(
            model, *inputs, **kwinputs
        )
        if jax.tree_structure(model) != model_treedef:
            raise ValueError("Expecting an updated model in the auxiliary output.")

        params = select_parameters(model)
        updates, optimizer = transform_gradients(grads, optimizer, params=params)
        params = apply_updates(params, updates=updates)
        model = update_parameters(model, params=params)
        return model, optimizer, aux

    def _update_fn_scan(
        model_and_optimizer: Union[C, Tuple[T, O]], *inputs, **kwinputs
    ) -> Tuple[C, Any]:
        model, optimizer = model_and_optimizer
        model, optimizer, aux = _update_fn(model, optimizer, *inputs, **kwinputs)
        return (model, optimizer), aux

    return _update_fn_scan if scan_mode else _update_fn


def scan(func, init, xs, length=None, unroll: int = 1, time_major=True):
    """``jax.lax.scan`` with an additional ``time_major=False`` mode.


    The semantics of ``scan`` are given roughly by this Python implementation:

    >>> def scan(f, init, xs, length=None):
    ...     if xs is None:
    ...         xs = [None] * length
    ...     carry = init
    ...     ys = []
    ...     for x in xs:
    ...         carry, y = f(carry, x)
    ...         ys.append(y)
    ...     return carry, np.stack(ys)
    """
    if time_major:
        # data format: TN...
        return jax.lax.scan(func, init, xs, length=length, unroll=unroll)
    else:
        # data format: NT...
        if xs is not None:
            xs = jnp.swapaxes(xs, 0, 1)  # swap batch and time axes
        state, output = jax.lax.scan(func, init, xs, length=length, unroll=unroll)
        output = jnp.swapaxes(output, 0, 1)  # restore to NT...
        return state, output
