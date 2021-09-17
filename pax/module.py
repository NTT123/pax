"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import jax.tree_util
import jmp
import numpy as np

from . import ctx

T = TypeVar("T", bound="Module")

# TODO: use NamedTuple, but, it is slower :-(
ModuleAuxiliaryData = Tuple

# All supported module's field kinds
class PaxFieldKind(Enum):
    """``PaxFieldKind`` lists all supported attribute kinds in ``pax.Module``.

    An attribute will be considered as part of the pytree structure if its kind is one of ``STATE_*``, ``PARAMETER_*``, ``MODULE_*``.

    * A ``STATE`` attribute is a non-trainable leaf of the pytree.
    * A ``STATE_SUBTREE`` attribute is a non-trainable subtree.
    * A ``PARAMETER`` attribute is a trainable leaf of the pytree.
    * A ``PARAMETER_SUBTREE`` attribute is a trainable subtree.
    * A ``MODULE`` or ``MODULE_SUBTREE`` attribute is a generic subtree.
    """

    STATE: int = 1
    PARAMETER: int = 2
    MODULE: int = 3
    STATE_SUBTREE: int = 4
    PARAMETER_SUBTREE: int = 5
    MODULE_SUBTREE: int = 6
    OTHERS: int = -1


class Module:
    """Module is the central object of Pax.

    It manages all information related to the pytree.
    It also includes methods (usually, ``__call__``) that can be executed to compute functions on the pytree.

    The two important methods: ``tree_flatten`` and ``tree_unflatten`` specify how a module can be converted to a ``(leaves, treedef)``,
    and otherwise from ``(treedef, leaves)`` back to a module.

    A module maintains a ``_name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    # Field Name To Kind
    _name_to_kind: Optional[Dict[str, PaxFieldKind]] = None
    _name_to_kind_to_unfreeze: Optional[Dict[str, PaxFieldKind]]
    _training: bool
    name: str

    def __new__(cls, *args, **kwargs):
        """Initialize _name_to_kind and _training in `__new__` method
        to avoid calling ``super().__init__()`` in the every subclass of Module."""
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot create new module in immutable mode")

        obj = object.__new__(cls)
        obj.__dict__.update(
            [
                ("_name_to_kind", MappingProxyType(OrderedDict())),
                ("_name_to_kind_to_unfreeze", None),
                ("_training", True),
                ("name", None),
            ]
        )

        return obj

    def __init__(self, name: Optional[str] = None):
        """Initialize module's name."""
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot create new module in immutable mode")

        super().__setattr__("name", name)

    @property
    def training(self) -> bool:
        return self._training

    def _update_name_to_kind_dict(self, name: str, value):
        """Update the `_name_to_kind` dictionary.

        Create a new dictionary and wrap it with
        `MappingProxyType`to avoid side effects."""
        if not ctx.state._enable_mutability:
            raise ValueError(
                "Cannot update `_name_to_kind` dictionary in immutable mode."
            )

        new_dict = OrderedDict(self._name_to_kind)
        new_dict[name] = value
        super().__setattr__("_name_to_kind", MappingProxyType(new_dict))

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Whenever a user sets ``value`` to attribute ``name``, we will check the assignment:

        * Setting ``_name_to_kind`` and ``_training`` are forbidden.

        * Setting ``value`` to a wrong kind attribute is also forbidden.

        * In `immutable` mode, only STATE and MODULE kinds are allowed to be set.

        * With `mutable` mode, all kinds are allowed to be set.

        * If ``value`` is a ``Module``'s instance and ``name`` is not in ``_name_to_kind``, its kind will be ``PaxFieldKind.MODULE``.
        """

        if name in ["_name_to_kind", "_training", "_name_to_kind_to_unfreeze"]:
            raise ValueError(
                f"You SHOULD NOT modify `{name}`. "
                f"If you _really_ want to, use `self.__dict__[name] = value` instead."
            )

        kind = self._name_to_kind.get(name, PaxFieldKind.OTHERS)

        leaves = jax.tree_flatten(value, is_leaf=lambda x: isinstance(x, Module))[0]
        ndarray_leaves = jax.tree_flatten(value)[0]
        all_modules = all(isinstance(mod, Module) for mod in leaves)

        if (
            len(ndarray_leaves) == 0
            and kind == PaxFieldKind.OTHERS
            and value is not None
        ):
            raise ValueError(
                f"Cannot assign an empty pytree of value `{value}` to an attribute of a Pax's Module."
            )

        from jax.dtypes import issubdtype as isdt

        def is_ndarray(x):
            return isinstance(x, jnp.ndarray)

        def is_differentable(x):
            return is_ndarray(x) and (
                isdt(x.dtype, jnp.complexfloating) or isdt(x.dtype, jnp.floating)
            )

        all_differentable = all(is_differentable(x) for x in ndarray_leaves)
        all_ndarray = all(is_ndarray(x) for x in ndarray_leaves)

        if kind != PaxFieldKind.OTHERS:
            if kind in [PaxFieldKind.PARAMETER, PaxFieldKind.STATE]:
                if not is_ndarray(value):
                    raise ValueError(
                        f"Assigning a non-ndarray value to an attribute of kind {kind}"
                    )
            elif kind in [PaxFieldKind.PARAMETER_SUBTREE, PaxFieldKind.STATE_SUBTREE]:
                if not all_ndarray:
                    raise ValueError(
                        f"Assigning a value which contains a non-ndarray object to an attribute of kind {kind}"
                    )
            elif kind == PaxFieldKind.MODULE:
                if not isinstance(value, Module):
                    raise ValueError(
                        f"Assigning a non-Module object to an attribute of kind {kind}"
                    )
            elif kind == PaxFieldKind.MODULE_SUBTREE:
                if not all_modules:
                    raise ValueError(
                        f"Assigning a value which contains a non-Module object to an attribute of kind {kind}"
                    )

        if ctx.state._enable_mutability:
            super().__setattr__(name, value)
        else:
            if kind in [
                PaxFieldKind.STATE,
                PaxFieldKind.STATE_SUBTREE,
            ]:
                super().__setattr__(name, value)
            else:
                raise ValueError(
                    f"Cannot set an attribute of kind `{kind}` in immutable mode."
                )

        # The automatic kind registering system:
        #   - if a value is a Module's instance, it is registered as Module,
        #   - if it contains Module's instances only, it is registered  Module_SUBTREE,
        #   - if it is differentiable ndarray,  it is registered as PARAMETER,
        #   - if it contains differentiable ndarray's only, it is registered as PARAMETER_SUBTREE,
        #   - if it is a non-differentiable ndarray,  it is registered as STATE,
        #   - if it contains non-differentiable ndarray's only, it is registered as STATE_SUBTREE.
        if name not in self._name_to_kind and value is not None:
            if isinstance(value, Module):
                self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)
            elif all_modules:
                self._update_name_to_kind_dict(name, PaxFieldKind.MODULE_SUBTREE)
            elif is_differentable(value):
                self._update_name_to_kind_dict(name, PaxFieldKind.PARAMETER)
            elif all_differentable:
                self._update_name_to_kind_dict(name, PaxFieldKind.PARAMETER_SUBTREE)
            elif is_ndarray(value):
                self._update_name_to_kind_dict(name, PaxFieldKind.STATE)
            elif all_ndarray:
                self._update_name_to_kind_dict(name, PaxFieldKind.STATE_SUBTREE)
            else:
                pass

        self._scan_fields(fields=(name,))

    def get_kind(self, name: str) -> PaxFieldKind:
        return self._name_to_kind[name]

    def set_kind(self, name: str, value: PaxFieldKind):
        self._name_to_kind[name] = value

    def __delattr__(self, name: str) -> None:
        if ctx.state._enable_mutability:
            super().__delattr__(name)
        else:
            raise ValueError(
                "Cannot delete module's attribute {name} in immutable mode."
            )

    def register_parameter(self, name: str, value: Optional[jnp.ndarray] = None):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.PARAMETER)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def register_state(self, name: str, value: Optional[jnp.ndarray] = None):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.STATE)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def register_module(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def register_parameter_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.PARAMETER_SUBTREE)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def register_state_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.STATE_SUBTREE)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def register_module_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._update_name_to_kind_dict(name, PaxFieldKind.MODULE_SUBTREE)

        if value is not None:
            setattr(self, name, value)
        elif not hasattr(self, name):
            raise ValueError("Cannot create a `None` attribute.")

    def tree_flatten(self):
        """Convert a module to ``(children, treedef)``."""
        fields = vars(self)

        children_names = []
        children = []
        not_tree = {}
        name_to_kind = self._name_to_kind

        for name, value in fields.items():
            if name in name_to_kind:
                children_names.append(name)
                children.append(value)
            else:
                not_tree[name] = value

        return children, (children_names, not_tree)

    @classmethod
    def tree_unflatten(cls, aux_data: ModuleAuxiliaryData, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        children_names, _not_tree = aux_data
        md = module.__dict__
        md.update(_not_tree)
        # don't have to copy `_name_to_kind` anymore, speed thing up!
        # md["_name_to_kind"] = OrderedDict(module._name_to_kind)
        md.update(zip(children_names, children))

        return module

    def __init_subclass__(cls):
        """Make sure any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        """Return a copy of current module."""
        return jax.tree_map(lambda x: x, self)

    def filter(self: T, keep: str = "parameter") -> T:
        """Filtering a module by trainable parameters and non-trainable states.

        Arguments:
            keep: type of leaves that will be kept ("parameter" or "state").
        """
        assert keep in ["parameter", "state"]
        fields = vars(self)
        cls = self.__class__
        module = object.__new__(cls)

        for name, value in fields.items():
            field_type = self._name_to_kind.get(name, PaxFieldKind.OTHERS)
            if field_type in [PaxFieldKind.MODULE, PaxFieldKind.MODULE_SUBTREE]:
                value = jax.tree_map(
                    lambda x: x.filter(keep),
                    value,
                    is_leaf=lambda x: isinstance(x, Module),
                )
            elif field_type in [PaxFieldKind.PARAMETER, PaxFieldKind.PARAMETER_SUBTREE]:
                fn1 = lambda x: x
                fn2 = lambda x: None
                fn = fn1 if keep == "parameter" else fn2
                value = jax.tree_map(fn, value)
            elif field_type in [PaxFieldKind.STATE, PaxFieldKind.STATE_SUBTREE]:
                fn1 = lambda x: x
                fn2 = lambda x: None
                fn = fn1 if keep == "state" else fn2
                value = jax.tree_map(fn, value)
            elif field_type == PaxFieldKind.OTHERS:
                pass
            else:
                raise ValueError("Not expected this!")
            module.__dict__[name] = value

        return module

    def update(self: T, other: T, in_place: bool = False) -> T:
        """Use parameters/state from ``other``.

        Arguments:
            other: parameter/state tree.
            in_place: modify the ``self`` object instead of copying it.
        """
        new_self = jax.tree_map(lambda s, o: (s if o is None else o), self, other)
        if in_place:
            self.__dict__.update(new_self.__dict__)
            return self
        else:
            return new_self

    def train(self: T, mode: bool = True):
        """Rebuild a new module recursively and set ``self._training = mode``.

        Arguments:
            mode: return a copy module in ``train`` mode module if ``True``.
        """
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot modify `_training` in immutable mode.")

        def _train_apply_fn(mod: T) -> T:
            mod.__dict__["_training"] = mode
            return mod

        return self.apply(_train_apply_fn)

    def eval(self: T) -> T:
        """Return a copy module in ``eval`` mode."""
        return self.train(False)

    def parameters(self):
        """Return trainable parameters of the module."""
        params = self.filter("parameter")
        return params

    def freeze(self: T) -> T:
        """Return a copy module with all trainable parameters are converted to non-trainable states."""
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot freeze a module in immutable mode.")

        def _freeze_fn(mod: T) -> T:
            if mod._name_to_kind_to_unfreeze is not None:
                raise ValueError("Freezing a frozen module is NOT allowed")

            new_name_to_kind = OrderedDict()
            for k, v in mod._name_to_kind.items():
                if v == PaxFieldKind.PARAMETER:
                    new_name_to_kind[k] = PaxFieldKind.STATE
                elif v == PaxFieldKind.PARAMETER_SUBTREE:
                    new_name_to_kind[k] = PaxFieldKind.STATE_SUBTREE
                else:
                    new_name_to_kind[k] = v

            # save a backup for later unfreeze calls
            mod.__dict__["_name_to_kind_to_unfreeze"] = mod.__dict__["_name_to_kind"]
            # use proxy to avoid any side effects
            mod.__dict__["_name_to_kind"] = MappingProxyType(new_name_to_kind)
            return mod

        return self.apply(_freeze_fn)

    def unfreeze(self: T) -> T:
        """Return the original module before frozen."""
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot unfreeze a module in immutable mode.")

        def _unfreeze_fn(mod: T) -> T:
            if mod._name_to_kind_to_unfreeze is None:
                return mod
            else:
                mod.__dict__["_name_to_kind"] = mod.__dict__[
                    "_name_to_kind_to_unfreeze"
                ]
                mod.__dict__["_name_to_kind_to_unfreeze"] = None
                return mod

        return self.apply(_unfreeze_fn)

    def hk_init(self, *args, enable_jit: bool = False, **kwargs):
        """Return a new initialized module.

        **Note**: This function is only useful if a module is/includes a converted module from the haiku library.

        Arguments:
            args, kwargs: dummy inputs to the module.
            enable_jit: to use `jax.jit` for the init function.
        """
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot call hk_init in immutable mode.")

        def init_fn(mod, args, kwargs):
            mod = mod.copy()
            mod(*args, **kwargs)
            return mod

        if enable_jit:
            init_fn = jax.jit(init_fn)
        return init_fn(self, args, kwargs)

    def sub_modules(self):
        """Return a list of sub-modules."""
        module_subtrees = [
            getattr(self, name)
            for name, kind in self._name_to_kind.items()
            if kind in [PaxFieldKind.MODULE, PaxFieldKind.MODULE_SUBTREE]
        ]

        submods, _ = jax.tree_flatten(
            module_subtrees, is_leaf=lambda x: isinstance(x, Module)
        )
        return [module for module in submods if isinstance(module, Module)]

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        """This is the default summary method.

        A module can customize its summary by overriding this method.

        Arguments:
            return_list: return a list of lines instead of a joined string.


        Example:
            >>> print(pax.nn.Sequential(pax.nn.Linear(2, 3), jax.nn.relu, pax.nn.Linear(3, 4)).summary())
                Sequential
                ├── Linear[in_dim=2, out_dim=3, with_bias=True]
                ├── x => relu(x)
                └── Linear[in_dim=3, out_dim=4, with_bias=True]
        """

        output = [self.__repr__()]
        if output[0] is None:
            raise ValueError(
                f"The `{self.__class__}.__repr__` method returns a `None` value."
            )
        sub_modules = self.sub_modules()

        def indent(lines: List[str], s) -> List[str]:
            return [s + l for l in lines]

        for i, module in enumerate(sub_modules):
            lines = module.summary(return_list=True)
            if i + 1 < len(sub_modules):  # middle submodules
                indented_lines = indent(lines[:1], "├── ") + indent(lines[1:], "│   ")
            else:  # last submodule
                indented_lines = indent(lines[:1], "└── ") + indent(lines[1:], "    ")
            output.extend(indented_lines)
        if return_list:
            return output
        else:
            return "\n".join(output)

    def deep_scan(self):
        """Scan a module recursively to find any *potential* bug."""

        fields = vars(self).keys()
        self._scan_fields(fields)

        for mod in self.sub_modules():
            mod.deep_scan()

    def _scan_fields(self, fields: Sequence[str]):
        """Scan fields for *potential* bugs."""

        from jax.dtypes import issubdtype as isdt

        for name in fields:
            value = getattr(self, name)
            kind = self._name_to_kind.get(name, PaxFieldKind.OTHERS)
            leaves = jax.tree_leaves(value)
            # Check if a parameter or parameter subtree
            # contains non-differentiable ndarray (e.g., uint32 array)
            if kind in [PaxFieldKind.PARAMETER, PaxFieldKind.PARAMETER_SUBTREE]:
                for leaf in leaves:
                    if hasattr(leaf, "dtype") and not (
                        isdt(leaf.dtype, jnp.complexfloating)
                        or isdt(leaf.dtype, jnp.floating)
                    ):
                        raise ValueError(
                            f"Field ``{self}.{name}`` of kind `{kind.name}` contains a non-differentiable leaf "
                            f"(type={leaf.dtype}, value={leaf})."
                        )

            # Check if a field contains unregistered ndarray
            if kind == PaxFieldKind.OTHERS:
                for leaf in leaves:
                    if isinstance(leaf, jnp.ndarray):
                        raise ValueError(
                            f"Unregistered field ``{self}.{name}`` of kind `{kind.name}` contains a ndarray "
                            f"(type={leaf.dtype}, value={leaf}). "
                            f"This is usually not a good idea. "
                            f"Consider registering it as a STATE or STATE_SUBTREE."
                        )

            # Check if an unregistered field contains pax.Module instance
            if kind not in [
                PaxFieldKind.MODULE,
                PaxFieldKind.MODULE_SUBTREE,
                PaxFieldKind.STATE,
                PaxFieldKind.STATE_SUBTREE,
            ]:
                mods, _ = jax.tree_flatten(
                    value, is_leaf=lambda x: isinstance(x, Module)
                )
                for mod in mods:
                    if isinstance(mod, Module):
                        raise ValueError(
                            f"Field ``{self}.{name}`` of kind `{kind.name}` "
                            f"SHOULD NOT contains a pax.Module instance: {mod}"
                        )

    def mixed_precision(self: T, mp_policy: jmp.Policy, method_name="__call__") -> T:
        """Convert the module to a MixedPrecision module.

        Return a clone object whose ``method_name`` method is wrapped to enforce the mixed-precision policy.

        Arguments:
            mp_policy: a ``jmp`` mixed precision policy.
            method_name: name of the method that will be affected.
        """
        if not ctx.state._enable_mutability:
            raise ValueError("Cannot apply mixed-precision policy in immutable mode.")

        if hasattr(self, "unwrap_mixed_precision"):
            raise ValueError(
                "Enforcing mixed-precision policy on an object twice is not allowed. "
                "Unwrap it with the `unwrap_mixed_precision` method first!"
            )
        casted_self = mp_policy.cast_to_param(self)

        cls = casted_self.__class__

        class MixedPrecisionWrapper(cls):
            def unwrap_mixed_precision(self):
                """Recreate the original class.

                Note: No guarantee that the parameter/state's dtype will be the same
                as the original module.
                """
                back = cls.__new__(cls)
                back.__dict__.update(self.__dict__)
                return back

        def mp_call(self_: T, *args, **kwargs):
            """This method does four tasks:

            Task 1: It casts all parameters and arguments to the "compute" data type.
            Task 2: It calls the original method.
            Task 3: It casts all the parameters back to the "param" data type.
               However, if a parameter is NOT modified during the forward pass,
               the original parameter will be reused to avoid a `cast` operation.
            Task 4: It casts the output to the "output" data type.
            """
            old_self_clone = self_.copy()

            # task 1
            casted_self, casted_args, casted_kwargs = mp_policy.cast_to_compute(
                (self_, args, kwargs)
            )
            self_.update(casted_self, in_place=True)

            casted_self_clone = self_.copy()

            # task 2
            output = getattr(cls, method_name)(self_, *casted_args, **casted_kwargs)

            # task 3
            if jax.tree_structure(self_) != jax.tree_structure(old_self_clone):
                raise RuntimeError(
                    f"The module `{self_.__class__.__name__}` has its treedef modified during the forward pass. "
                    f"This is currently not supported for a mixed-precision module!"
                )

            def reuse_params_fn(updated_new, new, old):
                # reuse the original parameter if it is
                # NOT modified during the forward pass.
                if updated_new is new:
                    return old  # nothing change
                else:
                    return mp_policy.cast_to_param(updated_new)

            casted_to_param_self = jax.tree_map(
                reuse_params_fn, self_, casted_self_clone, old_self_clone
            )

            self_.update(casted_to_param_self, in_place=True)

            # task 4
            output = mp_policy.cast_to_output(output)
            return output

        setattr(MixedPrecisionWrapper, method_name, mp_call)
        MixedPrecisionWrapper.__name__ = self.__class__.__name__ + "@MixedPrecision"
        new = MixedPrecisionWrapper.__new__(MixedPrecisionWrapper)
        new.__dict__.update(casted_self.__dict__)
        return new

    def apply(self, apply_fn):
        """Apply a function to all sub-modules.

        **Note**: this function returns a transformed copy of the current module.

        Arguments:
            apply_fn: a function which inputs a module and outputs a transformed module.
        """

        def rec_fn(x):
            if isinstance(x, Module):
                return x.apply(apply_fn)
            else:
                return x

        new_self = jax.tree_map(
            rec_fn, self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        # tree_map already created a copy of self,
        # hence `apply_fn` is guaranteed to have no side effects.
        return apply_fn(new_self)

    def __repr__(self, info: Optional[Dict[str, Any]] = None) -> str:
        name = f"({self.name}) " if self.name is not None else ""
        cls_name = self.__class__.__name__
        if info is None:
            return f"{name}{cls_name}"
        else:
            info = [f"{k}={v}" for (k, v) in info.items() if v is not None]
            info = ", ".join(info)
            return f"{name}{cls_name}[{info}]"
