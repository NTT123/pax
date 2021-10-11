"""PAX BaseModule.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

import threading
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax.dtypes import issubdtype as isdt

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")
TreeDef = Any


STATE = threading.local()
STATE.enable_deep_copy = False
STATE.inside_pure_function = False
STATE.mutable_module_list = ()


class enable_deep_copy(object):
    r"""A context manager that turns on deepcopy mode."""
    prev: Any

    def __init__(self):
        super().__init__()
        self.prev = STATE.enable_deep_copy

    def __enter__(self):
        self.prev = STATE.enable_deep_copy
        STATE.enable_deep_copy = True

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        STATE.enable_deep_copy = self.prev


class allow_mutation(object):
    r"""A context manager that turns on mutability."""
    prev: Any
    prev_inside: bool

    def __init__(self, modules):
        super().__init__()
        if isinstance(modules, BaseModule):
            modules = (modules,)
        self.mods = tuple(modules)

    def __enter__(self):
        self.prev = STATE.mutable_module_list
        STATE.mutable_module_list = self.mods + STATE.mutable_module_list
        self.prev_inside = STATE.inside_pure_function
        STATE.inside_pure_function = True

    def __exit__(self, _: Any, __: Any, __: Any) -> None:
        STATE.mutable_module_list = self.prev
        STATE.inside_pure_function = self.prev_inside


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


class PaxModuleInfo(NamedTuple):
    """PAX Internal Data Structure."""

    name: Optional[str]
    training: bool
    name_to_kind: Mapping[str, PaxFieldKind]

    def __repr__(self) -> str:
        nodes = ", ".join([f"{k}:{v.name}" for k, v in self.name_to_kind.items()])
        return f"PaxModuleInfo[name={self.name}, training={self.training}, nodes={{{nodes}}}]"


class ModuleMetaclass(type):
    """Metaclass for `Module`."""

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        module = cls.__new__(cls, *args, **kwargs)  # type: ignore

        # if a module is created inside a `pure` function, it is mutable.
        if STATE.inside_pure_function:
            STATE.mutable_module_list = (module,) + STATE.mutable_module_list
            cls.__init__(module, *args, **kwargs)
            module.find_and_register_submodules()
        else:
            with allow_mutation(module):
                cls.__init__(module, *args, **kwargs)
                module.find_and_register_submodules()

        # scan module after initialization for potential bugs
        module._scan_fields(module.__dict__)
        return module


class BaseModule(metaclass=ModuleMetaclass):
    """Module manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    Module maintains a ``_name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    _pax: PaxModuleInfo

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        """Initialize _name_to_kind and _training in `__new__` method to avoid
        calling `super().__init__()` in every subclass of Module."""

        del args, kwargs

        obj = object.__new__(cls)
        obj.__dict__["_pax"] = PaxModuleInfo(
            name=None,
            training=True,
            name_to_kind=MappingProxyType(OrderedDict()),
        )

        # scan class attributes for unregistered modules and ndarray's.
        obj._scan_fields(obj.__class__.__dict__)

        return obj

    def __init__(self, name: Optional[str] = None):
        """Initialize module's name."""
        super().__setattr__("_pax", self._pax._replace(name=name))

    def _assert_mutability(self):
        if id(self) not in [id(x) for x in STATE.mutable_module_list]:
            raise ValueError(
                "Cannot modify a module in immutable mode.\n"
                "Please do this computation inside a function decorated by `pax.pure`."
            )

    def _update_name_to_kind_dict(self, name: str, value):
        """Update the `_name_to_kind` dictionary.

        Create a new dictionary and wrap it with `MappingProxyType` to avoid side effects."""
        self._assert_mutability()

        new_dict = OrderedDict(self._pax.name_to_kind)
        new_dict[name] = value
        new_info = self._pax._replace(name_to_kind=MappingProxyType(new_dict))
        super().__setattr__("_pax", new_info)

    def __setattr__(self, name: str, value: Any) -> None:
        """Whenever a user sets an attribute, we will check the assignment:

        - Setting `_name_to_kind` and `_training` are forbidden.
        - If `value` is a pytree of modules and `name` is not in `_name_to_kind`,
        its kind will be `PaxFieldKind.MODULE`.
        """
        self._assert_mutability()

        if name == "_pax":
            raise ValueError(
                f"{name} is a reserved attribute for PAX internal mechanisms."
            )

        super().__setattr__(name, value)

        # If `value` contains Module's instances only, it is registered as MODULE.
        module_leaves, _ = jax.tree_flatten(
            value, is_leaf=lambda x: isinstance(x, BaseModule)
        )
        all_modules = all(isinstance(mod, BaseModule) for mod in module_leaves)
        if (
            value is not None
            and len(module_leaves) > 0
            and name not in self._pax.name_to_kind
            and (isinstance(value, BaseModule) or all_modules)
        ):
            self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)

        # scan the new field for bugs.
        self._scan_fields(fields=(name,))

    def __delattr__(self, name: str) -> None:
        self._assert_mutability()
        if name in self._pax.name_to_kind:
            raise ValueError("Cannot delete a pytree attribute.")
        super().__delattr__(name)

    def tree_flatten(self) -> Tuple[List[jnp.ndarray], Mapping[str, Any]]:
        """Convert a module to ``(children, treedef)``."""

        aux = dict(self.__dict__)
        children = [aux.pop(name) for name in self._pax.name_to_kind]

        if STATE.enable_deep_copy:
            leaves, treedef = jax.tree_flatten(aux)
            new_leaves = []
            for leaf in leaves:
                try:
                    new_leaf = deepcopy(leaf)
                    if new_leaf == leaf:
                        new_leaves.append(new_leaf)
                    else:
                        new_leaves.append(leaf)
                except:
                    new_leaves.append(leaf)

            aux = jax.tree_unflatten(treedef, leaves)

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        md = module.__dict__
        md.update(aux)
        # don't have to copy `_name_to_kind` anymore, speed thing up!
        # md["_name_to_kind"] = OrderedDict(module._name_to_kind)
        md.update(zip(module._pax.name_to_kind, children))

        # if a module is created inside a `pure` function, it is mutable.
        if STATE.inside_pure_function:
            STATE.mutable_module_list = (module,) + STATE.mutable_module_list

        return module

    def __init_subclass__(cls):
        """Any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def _scan_fields(self, fields: Sequence[str]):
        """Scan fields for *potential* bugs."""

        for name in fields:
            value = getattr(self, name)
            kind = self._pax.name_to_kind.get(name, PaxFieldKind.OTHERS)
            mods, _ = jax.tree_flatten(
                value, is_leaf=lambda x: isinstance(x, BaseModule)
            )
            leaves = jax.tree_leaves(value)

            # Check if a MODULE attribute contains non-module leafs.
            if kind == PaxFieldKind.MODULE:
                for mod in mods:
                    if not isinstance(mod, BaseModule):
                        raise ValueError(
                            f"Field `{self}.{name}` (kind={kind}, value={value}) contains a non-module leaf "
                            f"(type={type(mod)}, value={mod})."
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
                    value, is_leaf=lambda x: isinstance(x, BaseModule)
                )

                # Check if a field contains unregistered module
                for leaf in mods:
                    if isinstance(leaf, BaseModule):
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
                    if isinstance(mod, BaseModule):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) "
                            f"SHOULD NOT contains a pax.Module instance {mod}."
                        )

    def __eq__(self, o: object) -> bool:
        """Compare two modules."""
        if id(self) == id(o):
            return True

        if type(self) is not type(o):
            return False

        self_leaves, self_treedef = jax.tree_flatten(self)
        o_leaves, o_treedef = jax.tree_flatten(o)

        if len(self_leaves) != len(o_leaves):
            return False

        if self_treedef != o_treedef:
            return False

        leaves_equal = jax.tree_map(lambda a, b: a is b, self_leaves, o_leaves)
        return all(leaves_equal)

    def __hash__(self) -> int:
        leaves, treedef = jax.tree_flatten(self)
        leaves = jax.tree_map(lambda x: (x.shape, x.dtype), leaves)
        return hash((tuple(leaves), treedef))

    def find_and_register_submodules(self):
        """Find unregistered submodules and register it with MODULE kind."""

        def all_module_leaves(x):
            leaves = jax.tree_flatten(x, is_leaf=lambda m: isinstance(m, BaseModule))[0]
            return len(leaves) > 0 and all(isinstance(m, BaseModule) for m in leaves)

        for name, value in vars(self).items():
            if name not in self._pax.name_to_kind:
                if all_module_leaves(value):
                    self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)

    def register_subtree(self, name: str, value, kind: PaxFieldKind):
        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        self._update_name_to_kind_dict(name, kind)
        setattr(self, name, value)
