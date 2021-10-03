"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jmp

from . import ctx
from .module import Module, PaxFieldKind
from .utils import EmptyNode

TreeDef = Any

T = TypeVar("T", bound=Module)
K = TypeVar("K", bound=Module)
O = TypeVar("O", bound=Module)


def forward(mod: T, *inputs, params=None, **kwinputs) -> Tuple[T, Any]:
    """Execute the forward pass and return the updated module.

    Arguments:
        mod: The module to be executed.
        params: Use parameters in `params` if not ``None``.
    """
    assert callable(mod), "Expecting a callable module." ""
    mod = mod.copy()
    if params is not None:
        mod = update_parameters(mod, params=params)

    output = mod(*inputs, **kwinputs)
    return mod, output


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""

    def _train_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = True
        return mod

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation mode."""

    def _eval_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = False
        return mod

    return mod.apply(_eval_apply_fn)


def freeze_parameters(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        new_name_to_kind = OrderedDict()
        for k, v in mod._name_to_kind.items():
            if v == PaxFieldKind.PARAMETER:
                new_name_to_kind[k] = PaxFieldKind.STATE
            else:
                new_name_to_kind[k] = v

        # use proxy to avoid any side effects
        mod.__dict__["_name_to_kind"] = MappingProxyType(new_name_to_kind)
        return mod

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameters(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    assert freeze_parameters(origin) == mod
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_kind(mod: T, *, kind: PaxFieldKind) -> T:
    """Select leaves of kind ``kind`` while setting all other leaves to ``None``.

    Arguments:
        mod: The module.
        kind: The kind of leaves that will be kept intact.
    """
    assert kind in [PaxFieldKind.PARAMETER, PaxFieldKind.STATE]
    if kind == PaxFieldKind.STATE:
        none_list = [PaxFieldKind.PARAMETER]
    else:
        none_list = [PaxFieldKind.STATE]

    def _select_apply_fn(mod: T) -> T:
        for k, v in mod._name_to_kind.items():
            if v in none_list:
                value = getattr(mod, k)
                none_v = jax.tree_map(lambda _: EmptyNode(), value)
                with ctx.mutable():
                    setattr(mod, k, none_v)
        return mod

    return mod.apply(_select_apply_fn)


def select_parameters(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.PARAMETER)


def select_states(mod: T) -> T:
    """Select `STATE` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.STATE)


def scan_bugs(mod: T) -> T:
    """Scan the module for potential bugs."""

    def _scan_apply_fn(mod: T) -> T:
        assert isinstance(mod, Module)
        mod._scan_fields(mod.__class__.__dict__)
        mod._scan_fields(mod.__dict__)
        return mod

    return mod.apply(_scan_apply_fn)


def transform_gradients(grads: T, optimizer: O, *, params: T) -> Tuple[T, O]:
    """Transform gradients to updates using an optimizer.

    Arguments:
        grads: The gradients.
        optimizer: The gradient transformation.
        params: The trainable parameters.

    Returns:
        A pair ``(updates, optimizer)``

        - **updates** : The transformed gradients.
        - **optimizer** : The *updated* optimizer.
    """
    assert callable(optimizer), "Expecting a callable optimizer." ""
    optimizer = optimizer.copy()
    updates = optimizer(grads.parameters(), params=params)
    return updates, optimizer


def apply_updates(params: T, *, updates: T) -> T:
    """Update the parameters with updates.

    Arguments:
        params: The trainable parameters.
        updates: The transformed gradients.
    """
    from .utils import assertStructureEqual

    assertStructureEqual(updates, params)
    return jax.tree_map(lambda u, p: p - u, updates, params)


def apply_gradients(
    model: T, optimizer: K, *, grads: T, all_finite: Optional[jnp.ndarray] = None
) -> Tuple[T, K]:
    """Update model and optimizer with gradients `grads`.

    Arguments:
        model: the model which contains trainable parameters.
        optimizer: the gradient transformation.
        grads: the gradients w.r.t to trainable parameters of `model`.
        all_finite: True if gradients are finite. Default: `None`.

    Returns:
        A pair ``(new_model, new_optimizer)``

        - **new_model**: the updated model.
        - **new_optimizer**: the updated optimizer.
    """
    params = model.parameters()
    updates, new_optimizer = transform_gradients(grads, optimizer, params=params)
    new_params = apply_updates(params, updates=updates)

    if all_finite is not None:
        new_params, new_optimizer = jmp.select_tree(
            all_finite, (new_params, new_optimizer), (params, optimizer)
        )

    new_model = update_parameters(model, params=new_params)
    return new_model, new_optimizer


def update_pytree(mod: T, *, other: T) -> T:
    """Use non-EmptyNode leaves from others"""

    def _select_fn(x, y):
        if isinstance(y, EmptyNode):
            return x
        else:
            return y

    return jax.tree_map(
        _select_fn,
        mod,
        other,
    )


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module which uses trainable parameters in `params`."""
    return update_pytree(mod, other=select_parameters(params))


def update_states(mod: T, *, states: T) -> T:
    """Return a module which uses non-trainable states in `states`."""
    return update_pytree(mod, other=select_states(states))


def mutate(mod: T, *, with_fn: Callable[[T], K]) -> K:
    """Mutate a module without side effects and bugs."""
    with ctx.immutable():
        mod = mod.copy()  # prevent side effects
    mod = scan_bugs(mod)
    with ctx.mutable():
        new_mod = with_fn(mod)
        new_mod.find_and_register_submodules()
    new_mod = scan_bugs(new_mod)
    return new_mod


from .flatten_module import flatten_module
from .mixed_precision import apply_mp_policy
