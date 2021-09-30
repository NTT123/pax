"""Transform a module to a new one."""
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jmp

from . import ctx
from .module import Module, PaxFieldKind

TreeDef = Any

T = TypeVar("T", bound=Module)
K = TypeVar("K", bound=Module)


GradientTransformation = Module
O = TypeVar("O", bound=GradientTransformation)


def forward(mod: T, *inputs, params=None, **kwinputs) -> Tuple[T, Any]:
    """Execute the forward pass and return the updated module.

    Arguments:
        mod: The module to be executed.
        params: Use parameters in `params` if not ``None``.
    """
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
            elif v == PaxFieldKind.PARAMETER_SUBTREE:
                new_name_to_kind[k] = PaxFieldKind.STATE_SUBTREE
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


class flatten_module(Module, Generic[T]):
    """Flatten a module.

    Flatten all parameters and states to lists of `ndarray`'s."""

    from jaxlib.xla_extension import PyTreeDef

    params_leaves: List[jnp.ndarray]
    states_leaves: List[jnp.ndarray]
    params_treedef: PyTreeDef
    states_treedef: PyTreeDef
    module_treedef: PyTreeDef

    def __init__(self, mod: T):
        """Create a flatten version of the input module."""
        super().__init__()

        params_leaves, params_treedef = jax.tree_flatten(select_parameters(mod))
        states_leaves, states_treedef = jax.tree_flatten(select_states(mod))

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
            states_leaves, _ = jax.tree_flatten(select_states(module))
        self.states_leaves = states_leaves
        return out

    def __repr__(self) -> str:
        s = self.unflatten().__repr__()
        return f"Flatten({s})"


class apply_mp_policy(Module, Generic[T]):
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

        casted_mod_clone = casted_mod.copy()
        # task 2
        output = casted_mod(*casted_args, **casted_kwargs)

        # task 3
        if jax.tree_structure(casted_mod) != jax.tree_structure(old_mod_clone):
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

        self._module = jax.tree_map(
            reuse_params_fn, casted_mod, casted_mod_clone, old_mod_clone
        )

        # task 4
        output = self.mp_policy.cast_to_output(output)
        return output

    def __repr__(self):
        dtype_to_name = {
            jnp.bfloat16: "bfloat16",
            jnp.float16: "float16",
            jnp.float32: "float32",
            jnp.float64: "float64",
        }

        info = {
            "param_dtype": dtype_to_name[self.mp_policy.param_dtype],
            "compute_dtype": dtype_to_name[self.mp_policy.compute_dtype],
            "output_dtype": dtype_to_name[self.mp_policy.output_dtype],
        }
        return super().__repr__(info=info)


def update_parameters(mod: T, *, params: T) -> T:
    """Return a module which uses trainable parameters in `params`."""
    return mod.update(select_parameters(params))


def update_states(mod: T, *, states: T) -> T:
    """Return a module which uses non-trainable states in `states`."""
    return mod.update(select_states(states))


def mutate(mod: T, *, with_fn: Callable[[T], K]) -> K:
    """Mutate a module without side effects and bugs."""
    with ctx.immutable():
        mod = mod.copy()  # prevent side effects
    mod = scan_bugs(mod)
    new_mod = with_fn(mod)
    new_mod = scan_bugs(new_mod)
    return new_mod
