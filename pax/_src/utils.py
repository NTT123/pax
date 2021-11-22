"""Useful functions."""

import functools
from typing import Any, Callable, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jmp

from .core import (
    Module,
    apply_mp_policy,
    module_and_value,
    select_parameters,
    update_parameters,
)
from .core.module import load_weights_from_dict, save_weights_to_dict
from .nn import (
    BatchNorm1D,
    BatchNorm2D,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    GroupNorm,
    LayerNorm,
    Linear,
)

GradientTransformation = Module
T = TypeVar("T", bound=Module)
O = TypeVar("O", bound=GradientTransformation)
C = TypeVar("C")
K = TypeVar("K")


def grad(
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
    ... def loss_fn(model: pax.Linear, x, y):
    ...     y_hat = model(x)
    ...     loss = jnp.mean(jnp.square(y - y_hat))
    ...     return loss, (loss, model)
    ...
    >>> grad_fn = pax.grad(loss_fn, has_aux=True)
    >>> net = pax.Linear(1, 1)
    >>> x = jnp.ones((3, 1))
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


def value_and_grad(func: Callable[..., K], has_aux=False):
    """A PAX-compatible version of jax.value_and_grad.

    This version computes gradients w.r.t. trainable parameters of a PAX module.
    """

    def func_with_params(params: T, module: T, *args, **kwargs):
        return func(module | params, *args, **kwargs)

    vag_fn = jax.value_and_grad(func_with_params, has_aux=has_aux)

    @functools.wraps(vag_fn)
    def new_vag_fn(module: T, *args, **kwargs) -> Tuple[K, T]:
        return vag_fn(module.parameters(), module, *args, **kwargs)

    return new_vag_fn


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
    >>> net = pax.Linear(2, 2)
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
        grads, (aux, model) = grad(loss_fn, has_aux=True)(model, *inputs, **kwinputs)
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
            # swap batch and time axes
            xs = jax.tree_map(lambda leaf: jnp.swapaxes(leaf, 0, 1), xs)
        state, output = jax.lax.scan(func, init, xs, length=length, unroll=unroll)
        # restore to format NT...
        output = jax.tree_map(lambda leaf: jnp.swapaxes(leaf, 0, 1), output)
        return state, output


def apply_scaled_gradients(model: T, optimizer: Callable, loss_scale, grads: T):
    """Update model, optimizer and loss scale.

    Example:

    >>> import jmp
    >>> from pax.experimental import apply_scaled_gradients
    >>> net = pax.Linear(2, 2)
    >>> opt = opax.adam(1e-4)(net.parameters())
    >>> loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2**15))
    >>> grads = net.parameters()
    >>> net, opt, loss_scale = apply_scaled_gradients(net, opt, loss_scale, grads)
    >>> print(loss_scale.loss_scale)
    32770.0
    """
    params = model.parameters()
    grads = loss_scale.unscale(grads)
    skip_nonfinite_updates = isinstance(loss_scale, jmp.DynamicLossScale)
    if skip_nonfinite_updates:
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        new_optimizer, updates = module_and_value(optimizer)(grads, params)
        new_params = params.map(jax.lax.sub, updates)
        new_model = model.update_parameters(new_params)
        model, optimizer = jmp.select_tree(
            grads_finite,
            (new_model, new_optimizer),
            (model, optimizer),
        )
    else:
        optimizer, updates = module_and_value(optimizer)(grads, params)
        params = params.map(jax.lax.sub, updates)
        model = model.update_parameters(params)
    return model, optimizer, loss_scale


def default_mp_policy(module: T) -> T:
    """A default mixed precision policy.

    - Linear layers are in half precision.
    - Normalization layers are in full precision.

    Example:

    >>> net = pax.Sequential(pax.Linear(3, 3), pax.BatchNorm1D(3))
    >>> net = net.apply(pax.experimental.default_mp_policy)
    >>> print(net.summary())
    Sequential
    ├── Linear(in_dim=3, out_dim=3, with_bias=True, mp_policy=FHF)
    └── BatchNorm1D(num_channels=3, ..., mp_policy=FFF)
    """
    half = jmp.half_dtype()
    full = jnp.float32
    linear_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
    norm_policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)
    linear_classes = (Linear, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose)
    norm_classes = (BatchNorm1D, BatchNorm2D, GroupNorm, LayerNorm)

    if isinstance(module, linear_classes):
        return apply_mp_policy(module, mp_policy=linear_policy)
    elif isinstance(module, norm_classes):
        return apply_mp_policy(module, mp_policy=norm_policy)
    else:
        return module  # unchanged
