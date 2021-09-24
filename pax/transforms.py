"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Callable, Generic, List, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jmp
from jaxlib.xla_extension import PyTreeDef

from . import ctx
from .pax_transforms import grad

TreeDef = Any
from .module import Module, PaxFieldKind

T = TypeVar("T", bound="Module")


GradientTransformation = Module


def enable_train_mode(mod: T) -> T:
    """Return a module in training mode."""

    def _train_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = True
        return mod

    return mod.apply(_train_apply_fn)


def enable_eval_mode(mod: T) -> T:
    """Return a module in evaluation model."""

    def _eval_apply_fn(mod: T) -> T:
        mod.__dict__["_training"] = False
        return mod

    return mod.apply(_eval_apply_fn)


def freeze_parameter(mod: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    def _freeze_apply_fn(mod: T) -> T:
        new_name_to_kind = OrderedDict()
        for k, v in mod._name_to_kind.items():
            if v == PaxFieldKind.PARAMETER:
                new_name_to_kind[k] = PaxFieldKind.STATE
            elif v == PaxFieldKind.PARAMETER_SUBTREE:
                new_name_to_kind[k] = PaxFieldKind.STATE_SUBTREE
            else:
                new_name_to_kind[k] = v

        # use proxy to avoid any side effects
        mod.__dict__["_name_to_kind"] = MappingProxyType(new_name_to_kind)
        return mod

    return mod.apply(_freeze_apply_fn)


def unfreeze_parameter(mod: T, *, origin: T) -> T:
    """Return a copy module with all trainable parameters are converted to non-trainable states."""

    assert freeze_parameter(origin) == mod
    tree_def = jax.tree_structure(origin)
    leaves = jax.tree_leaves(mod)
    return jax.tree_unflatten(tree_def, leaves)


def select_kind(mod: T, *, kind: PaxFieldKind) -> T:
    """Select leaves of kind `kind` while setting all other leaves to ``None``.

    Arguments:
        mod: The module.
        kind: The kind of leaves that will be kept intact.
    """
    assert kind in [PaxFieldKind.PARAMETER, PaxFieldKind.STATE]
    if kind == PaxFieldKind.STATE:
        none_list = [PaxFieldKind.PARAMETER, PaxFieldKind.PARAMETER_SUBTREE]
    else:
        none_list = [PaxFieldKind.STATE, PaxFieldKind.STATE_SUBTREE]

    def _select_apply_fn(mod: T) -> T:
        for k, v in mod._name_to_kind.items():
            if v in none_list:
                value = getattr(mod, k)
                none_v = jax.tree_map(lambda _: None, value)
                setattr(mod, k, none_v)
        return mod

    return mod.apply(_select_apply_fn)


def select_parameter(mod: T) -> T:
    """Select `PARAMETER` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.PARAMETER)


def select_state(mod: T) -> T:
    """Select `STATE` leaves only."""
    return select_kind(mod, kind=PaxFieldKind.STATE)


def scan_bug(mod: T) -> T:
    """Scan the module for potential bugs."""

    def _scan_apply_fn(mod: T) -> T:
        assert isinstance(mod, Module)
        mod._scan_fields(mod.__class__.__dict__)
        mod._scan_fields(mod.__dict__)
        return mod

    return mod.apply(_scan_apply_fn)


def transform_gradient(grads: T, *, params: T, optimizer: Module) -> Tuple[T, Module]:
    """Transform gradients to updates using an optimizer.

    Arguments:
        grads: The gradients.
        params: The trainable parameters.
        optimizer: The gradient transformation.

    Returns: (updates, optimizer)
        - **updates** : The transformed gradients.
        - **optimizer** : The *updated* optimizer.
    """
    optimizer = optimizer.copy()
    updates = optimizer(grads, params=params)
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


def grad_with_aux(model: T, *, fn: Callable, inputs: Any) -> Tuple[T, Any]:
    """Return the gradients of function ``fn`` with respect to ``model``'s trainable parameters.

    Arguments:
        model: The module which contains trainable parameters and forward-pass computation.
        fn: A loss function, whose inputs are `(params, model, inputs)`.
        inputs: The inputs to `fn`.

    Returns: (grads, aux)
        - **grads** : The gradients w.r.t. model's trainable parameters.
        - **aux** : The auxiliary information (usually `loss` and the updated `model`).
    """
    model = model.copy()  # prevent side effects
    grads, aux = grad(fn, has_aux=True)(select_parameter(model), model, inputs)

    return grads, aux


class flatten_module(Generic[T], Module):
    """Flatten a module.

    Flatten all parameters and states to lists of `ndarray`'s."""

    params_leaves: List[jnp.ndarray]
    states_leaves: List[jnp.ndarray]
    params_treedef: PyTreeDef
    states_treedef: PyTreeDef
    module_treedef: PyTreeDef

    def __init__(self, mod: T):
        """Create a flatten version of the input module."""
        super().__init__()

        params_leaves, params_treedef = jax.tree_flatten(select_parameter(mod))
        states_leaves, states_treedef = jax.tree_flatten(select_state(mod))

        self.params_treedef = params_treedef
        self.states_treedef = states_treedef
        self.module_treedef = jax.tree_structure(mod)
        self.register_parameter_subtree("params_leaves", params_leaves)
        self.register_state_subtree("states_leaves", states_leaves)
        self.num_leaves = len(jax.tree_leaves(mod))

        if hasattr(mod, "unflatten"):
            raise RuntimeError("Cannot flatten a module twice!")

        if not hasattr(mod, "__call__"):
            raise ValueError("Expecting a callable module.")

    def unflatten(self) -> T:
        """Recreate the original module."""
        params = jax.tree_unflatten(self.params_treedef, self.params_leaves)
        states = jax.tree_unflatten(self.states_treedef, self.states_leaves)
        module = jax.tree_unflatten(self.module_treedef, [0] * self.num_leaves)
        module = module.update(params)
        module = module.update(states)
        return module

    def __call__(self, *args, **kwargs):
        """Recreate the original module, then call it."""
        module = self.unflatten()
        out = module(*args, **kwargs)

        with ctx.mutable():
            states_leaves, _ = jax.tree_flatten(select_state(module))
        self.states_leaves = states_leaves
        return out

    def __repr__(self) -> str:
        s = self.unflatten().__repr__()
        return f"Flatten({s})"


class apply_mp_policy(Generic[T], Module):
    """Convert the module to a mixed-precision module."""

    _module: T

    def __init__(self, mod: T, *, mp_policy: jmp.Policy):
        """Create a wrapper module to enforce the mixed-precision policy.

        Arguments:
            mod: the module.
            mp_policy: a ``jmp`` mixed precision policy.
        """
        super().__init__()

        if hasattr(mod, "unwrap_mixed_precision"):
            raise ValueError(
                "Enforcing mixed-precision policy on an object twice is not allowed. "
                "Unwrap it with the `unwrap_mixed_precision` method first!"
            )

        self._module = mp_policy.cast_to_param(mod)
        self.mp_policy = mp_policy

        if not hasattr(mod, "__call__"):
            raise ValueError("Expecting a callable module.")

    def unwrap_mixed_precision(self) -> T:
        """Recreate the original module.

        **Note**: No guarantee that the parameter/state's dtype will be the same as the original module.
        """
        return self._module.copy()

    def __call__(self, *args, **kwargs):
        """This method does four tasks:

        * Task 1: It casts all parameters and arguments to the "compute" data type.
        * Task 2: It calls the original module.
        * Task 3: It casts all the parameters back to the "param" data type.
          However, if a parameter is NOT modified during the forward pass,
          the original parameter will be reused to avoid a `cast` operation.
        * Task 4: It casts the output to the "output" data type.

        """
        old_mod_clone = self._module.copy()

        # task 1
        casted_mod, casted_args, casted_kwargs = self.mp_policy.cast_to_compute(
            (self._module, args, kwargs)
        )

        self._module.update(casted_mod, in_place=True)

        casted_mod_clone = self._module.copy()
        # task 2
        output = self._module(*casted_args, **casted_kwargs)

        # task 3
        if jax.tree_structure(self._module) != jax.tree_structure(old_mod_clone):
            raise RuntimeError(
                f"The module `{self._module.__class__.__name__}` has its treedef modified during the forward pass. "
                f"This is currently not supported for a mixed-precision module!"
            )

        def reuse_params_fn(updated_new, new, old):
            # reuse the original parameter if it is
            # NOT modified during the forward pass.
            if updated_new is new:
                return old  # nothing change
            else:
                return self.mp_policy.cast_to_param(updated_new)

        casted_to_param_mod = jax.tree_map(
            reuse_params_fn, self._module, casted_mod_clone, old_mod_clone
        )

        self._module.update(casted_to_param_mod, in_place=True)

        # task 4
        output = self.mp_policy.cast_to_output(output)
        return output
