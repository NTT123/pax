"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from collections import OrderedDict
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax.dtypes import issubdtype as isdt

from .ctx import state as ctx_state

T = TypeVar("T", bound="Module")


class PaxFieldKind(Enum):
    """``PaxFieldKind`` lists all supported attribute kinds in ``pax.Module``.

    An attribute will be considered as part of the pytree structure
    if its kind is one of ``STATE``, ``PARAMETER``, ``MODULE``.

    * A ``STATE`` attribute is a non-trainable leaf of the pytree.
    * A ``PARAMETER`` attribute is a trainable leaf of the pytree.
    * A ``MODULE`` attribute is a generic subtree.
    """

    STATE: int = 1
    PARAMETER: int = 2
    MODULE: int = 3
    OTHERS: int = -1


class ModuleMetaclass(type):
    """Metaclass for `Module`."""

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        module = cls.__new__(cls, *args, **kwargs)
        cls.__init__(module, *args, **kwargs)
        module._find_and_register_submodules()
        # scan module after initialization for potential bugs
        module._scan_fields(module.__dict__)

        return module


class Module(object, metaclass=ModuleMetaclass):
    """Module manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    Module maintains a ``_name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    # Field Name To Kind
    _name_to_kind: Dict[str, PaxFieldKind]
    _training: bool
    _name: Optional[str]

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        """Initialize _name_to_kind and _training in `__new__` method to avoid
        calling `super().__init__()` in every subclass of Module."""

        if not ctx_state._enable_mutability:
            raise ValueError("Cannot create new module in immutable mode")

        obj = object.__new__(cls)
        obj.__dict__["_name_to_kind"] = MappingProxyType(OrderedDict())
        obj.__dict__["_training"] = True
        obj.__dict__["_name"] = None

        # scan class attributes for unregistered modules and ndarray's.
        obj._scan_fields(obj.__class__.__dict__)

        return obj

    def __init__(self, name: Optional[str] = None):
        """Initialize module's name."""
        super().__setattr__("_name", name)

    @property
    def training(self) -> bool:
        return self._training

    @property
    def name(self) -> Optional[str]:
        return self._name

    def _update_name_to_kind_dict(self, name: str, value):
        """Update the `_name_to_kind` dictionary.

        Create a new dictionary and wrap it with `MappingProxyType` to avoid side effects."""
        if not ctx_state._enable_mutability:
            raise ValueError(
                "Cannot update `_name_to_kind` dictionary in immutable mode."
            )

        new_dict = OrderedDict(self._name_to_kind)
        new_dict[name] = value
        super().__setattr__("_name_to_kind", MappingProxyType(new_dict))

    def __setattr__(self, name: str, value: Any) -> None:
        """Whenever a user sets an attribute, we will check the assignment:

        - Setting `_name_to_kind` and `_training` are forbidden.
        - In immutable mode, only STATE attributes are allowed to be set. In mutable mode, all kinds are allowed to be set.
        - If `value` is a pytree of modules and `name` is not in `_name_to_kind`, its kind will be `PaxFieldKind.MODULE`.
        """

        if name in ["_name_to_kind", "_training", "_name"]:
            raise ValueError(
                f"{name} is a reserved attribute for Pax internal mechanisms."
            )

        kind = self._name_to_kind.get(name, PaxFieldKind.OTHERS)

        if ctx_state._enable_mutability:
            super().__setattr__(name, value)
        else:
            if kind == PaxFieldKind.STATE:
                super().__setattr__(name, value)
            else:
                raise ValueError(
                    f"Cannot assign an attribute of kind `{kind}` in immutable mode."
                )

        # If `value` contains Module's instances only, it is registered as MODULE.
        module_leaves, _ = jax.tree_flatten(
            value, is_leaf=lambda x: isinstance(x, Module)
        )
        all_modules = all(isinstance(mod, Module) for mod in module_leaves)
        if (
            value is not None
            and len(module_leaves) > 0
            and name not in self._name_to_kind
            and (isinstance(value, Module) or all_modules)
        ):
            self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)

        self._scan_fields(fields=(name,))

    def __delattr__(self, name: str) -> None:
        if ctx_state._enable_mutability:
            if name in self._name_to_kind:
                raise ValueError("Cannot delete pytree attribute.")
            super().__delattr__(name)
        else:
            raise ValueError(
                "Cannot delete module's attribute {name} in immutable mode."
            )

    def register_parameter(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER`` in the ``_name_to_kind`` dictionary."""

        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        self._update_name_to_kind_dict(name, PaxFieldKind.PARAMETER)
        setattr(self, name, value)

    def register_state(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE`` in the ``_name_to_kind`` dictionary."""

        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        self._update_name_to_kind_dict(name, PaxFieldKind.STATE)
        setattr(self, name, value)

    def register_modules(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE`` in the ``_name_to_kind`` dictionary."""

        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)
        setattr(self, name, value)

    # TODO: this is redundant, fix it!
    register_parameters = register_parameter
    register_states = register_state
    register_module = register_modules

    def tree_flatten(self) -> Tuple[list, Tuple[List[str], Any]]:
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

        if not ctx_state._enable_mutability:
            leaves, treedef = jax.tree_flatten(not_tree)
            # TODO: it is possible that `leaves` can change its internal states,
            # `jax.jit` will not detect the change. Fix this!
            not_tree = jax.tree_unflatten(treedef, leaves)

        return children, (children_names, not_tree)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        children_names, not_tree = aux_data
        md = module.__dict__
        md.update(not_tree)
        # don't have to copy `_name_to_kind` anymore, speed thing up!
        # md["_name_to_kind"] = OrderedDict(module._name_to_kind)
        md.update(zip(children_names, children))

        return module

    def __init_subclass__(cls):
        """Any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        """Return a copy of current module."""
        return jax.tree_map(lambda x: x, self)

    def submodules(self) -> List["Module"]:
        """Return a list of submodules."""
        module_subtrees = [
            getattr(self, name)
            for name, kind in self._name_to_kind.items()
            if kind == PaxFieldKind.MODULE
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
        submodules = self.submodules()

        def indent(lines: List[str], s) -> List[str]:
            return [s + l for l in lines]

        for i, module in enumerate(submodules):
            lines = module.summary(return_list=True)
            if i + 1 < len(submodules):  # middle submodules
                indented_lines = indent(lines[:1], "├── ") + indent(lines[1:], "│   ")
            else:  # last submodule
                indented_lines = indent(lines[:1], "└── ") + indent(lines[1:], "    ")
            output.extend(indented_lines)
        if return_list:
            return output
        else:
            return "\n".join(output)

    def _scan_fields(self, fields: Sequence[str]):
        """Scan fields for *potential* bugs."""

        for name in fields:
            value = getattr(self, name)
            kind = self._name_to_kind.get(name, PaxFieldKind.OTHERS)
            mods, _ = jax.tree_flatten(value, is_leaf=lambda x: isinstance(x, Module))
            leaves = jax.tree_leaves(value)

            # Check if a MODULE attribute contains non-module leafs.
            if kind == PaxFieldKind.MODULE:
                for mod in mods:
                    if not isinstance(mod, Module):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) contains a non-module leaf "
                            f"(type={type(leaf)}, value={leaf})."
                        )

            # Check if a pytree attribute contains non-ndarray values.
            if kind != PaxFieldKind.OTHERS:
                for leaf in leaves:
                    if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) contains a non-ndarray value "
                            f"(type={type(leaf)}, value={leaf})."
                        )

            # Check if a PARAMETER attribute contains non-differentiable ndarray's.
            if kind == PaxFieldKind.PARAMETER:
                for leaf in leaves:
                    if hasattr(leaf, "dtype") and not (
                        isdt(leaf.dtype, jnp.complexfloating)
                        or isdt(leaf.dtype, jnp.floating)
                    ):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) contains a non-differentiable leaf "
                            f"(type={leaf.dtype}, value={leaf})."
                        )

            if kind == PaxFieldKind.OTHERS:
                # if a field contains empty pytree
                leaves, _ = jax.tree_flatten(
                    value, is_leaf=lambda x: isinstance(x, Module)
                )

                # Check if a field contains unregistered module
                for leaf in mods:
                    if isinstance(leaf, Module):
                        raise ValueError(
                            f"Unregistered field `{self}.{name}` ({kind}) contains a Module "
                            f"({leaf}). "
                            f"Consider registering it as a MODULE."
                        )

                # Check if a field contains unregistered ndarray
                for leaf in leaves:
                    if isinstance(leaf, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            f"Unregistered field `{self}.{name}` ({kind}) contains a ndarray. "
                            f"Consider registering it using `self.register_*` methods."
                        )

            # Check if an unregistered (or PARAMETER) field contains pax.Module instances
            if kind not in [PaxFieldKind.MODULE, PaxFieldKind.STATE]:
                for mod in mods:
                    if isinstance(mod, Module):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) "
                            f"SHOULD NOT contains a pax.Module instance {mod}."
                        )

    def apply(self: T, apply_fn) -> T:
        """Apply a function to all submodules.

        **Note**: this function returns a transformed copy of the module.

        Arguments:
            apply_fn: a function which inputs a module and outputs a transformed module.
        """

        def rec_fn(x):
            if isinstance(x, Module):
                return x.apply(apply_fn)
            else:
                return x

        submodules = self.submodules()
        new_self = jax.tree_map(
            rec_fn,
            self,
            is_leaf=lambda x: isinstance(x, Module) and (x in submodules),
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
            lst_info = [f"{k}={v}" for (k, v) in info.items() if v is not None]
            str_info = ", ".join(lst_info)
            return f"{name}{cls_name}[{str_info}]"

    def __eq__(self, o: object) -> bool:
        """Compare two modules."""
        self_leaves, self_treedef = jax.tree_flatten(self)
        o_leaves, o_treedef = jax.tree_flatten(o)
        if len(self_leaves) != len(o_leaves):
            return False
        elif self_treedef != o_treedef:
            return False
        else:
            leaves_equal = jax.tree_map(lambda a, b: a is b, self_leaves, o_leaves)
            return all(leaves_equal)

    def __hash__(self) -> int:
        leaves, treedef = jax.tree_flatten(self)
        leaves = jax.tree_map(lambda x: (x.shape, x.dtype), leaves)
        return hash((tuple(leaves), treedef))

    def train(self: T) -> T:
        """Return a module in training mode."""
        from .transforms import enable_train_mode

        return enable_train_mode(self)

    def eval(self: T) -> T:
        """Return a module in evaluation mode."""
        from .transforms import enable_eval_mode

        return enable_eval_mode(self)

    def parameters(self: T) -> T:
        """Return trainable parameters."""
        from .transforms import select_parameters

        return select_parameters(self)

    def forward(self, *args, params=None, **kwargs):
        from .transforms import forward

        return forward(self, *args, params=params, **kwargs)

    def update_parameters(self: T, params: T) -> T:
        from .transforms import update_parameters

        return update_parameters(self, params=params)

    def _find_and_register_submodules(self):
        """Find unregistered submodules and register it with MODULE kind."""

        def all_module_leaves(x):
            leaves = jax.tree_flatten(x, is_leaf=lambda m: isinstance(m, Module))[0]
            return len(leaves) > 0 and all(isinstance(m, Module) for m in leaves)

        for name, value in vars(self).items():
            if name not in self._name_to_kind:
                if all_module_leaves(value):
                    self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)
