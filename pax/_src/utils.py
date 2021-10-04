"""Useful functions."""

from typing import Any, Callable, Tuple, TypeVar, Union
from unittest import TestCase

import jax
import jax.numpy as jnp

from .module import Module, PaxFieldKind
from .rng import KeyArray
from .strict_mode import grad

GradientTransformation = Module
T = TypeVar("T", bound=Module)
O = TypeVar("O", bound=GradientTransformation)
C = TypeVar("C")


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
    """Compute gradient with respect to trainable parameters of the first argument.

    Example:

    >>> def loss_fn(model: pax.nn.Linear, x, y):
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

    _grad_fn = grad(_fun, has_aux=True, allow_int=False, io_check=True, copy=True)

    def grad_fn(mod: T, *args, **kwargs) -> Tuple[T, C]:
        if not isinstance(mod, Module):
            raise ValueError("Expecting a Pax's Module at the first argument.")

        out = _grad_fn(mod.parameters(), mod, *args, **kwargs)
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

    def _update_fn(model: T, optimizer: O, *inputs, **kwinputs) -> Tuple[T, O, Any]:
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

        from .transforms import (
            apply_updates,
            select_parameters,
            transform_gradients,
            update_parameters,
        )

        assert isinstance(model, Module)
        assert isinstance(optimizer, Module)

        model_treedef = jax.tree_structure(model)
        grads, (aux, model) = grad_parameters(loss_fn)(model, *inputs, **kwinputs)
        assert (
            jax.tree_structure(model) == model_treedef
        ), "Expecting an updated model in the auxiliary output."

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

    try:
        jax.tree_map(
            check,
            self,
            other,
            is_leaf=lambda x: isinstance(x, Module)
            and x is not self
            and x is not other,
        )
    except ValueError:
        tc = TestCase()
        tc.maxDiff = None
        tc.assertDictEqual(vars(self), vars(other))
