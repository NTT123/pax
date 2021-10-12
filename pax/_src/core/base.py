"""PAX BaseModule."""

# Note: This file is originated from
# https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
# which is under MIT License.

import threading
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax.dtypes import issubdtype as isdt

from .rng import get_rng_state, set_rng_state

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")
TreeDef = Any


STATE = threading.local()
STATE.enable_deep_copy = False
STATE.inside_pure_function = False
STATE.mutable_module_id_list = ()


@contextmanager
def enable_deep_copy():
    r"""A context manager that turns on deepcopy mode."""
    prev = STATE.enable_deep_copy
    STATE.enable_deep_copy = True
    try:
        yield
    finally:
        STATE.enable_deep_copy = prev


@contextmanager
def allow_mutation(modules):
    r"""A context manager that turns on mutability."""
    if not isinstance(modules, (tuple, list)):
        modules = (modules,)
    modules = tuple([id(mod) for mod in modules])

    prev = STATE.mutable_module_id_list
    prev_inside = STATE.inside_pure_function
    prev_rng_state = get_rng_state()
    try:
        STATE.inside_pure_function = True
        STATE.mutable_module_id_list = modules
        yield
    finally:
        STATE.mutable_module_id_list = prev
        STATE.inside_pure_function = prev_inside
        set_rng_state(prev_rng_state)


@jax.tree_util.register_pytree_node_class
class EmptyNode(Tuple):
    """We use this class to mark deleted nodes.

    Note: this is inspired by treex's `Nothing` class.
    """

    def tree_flatten(self):
        """Flatten empty node."""
        return (), None

    @classmethod
    def tree_unflatten(cls, _, __):
        """Unflatten empty node."""
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
            STATE.mutable_module_id_list = (id(module),) + STATE.mutable_module_id_list
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
    """BaseModule manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    BaseModule maintains a ``name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    _pax: PaxModuleInfo

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        """Initialize `_pax` attribute in `__new__` method to avoid
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
        if id(self) not in STATE.mutable_module_id_list:
            raise ValueError(
                "Cannot modify a module in immutable mode.\n"
                "Please do this computation inside a function decorated by `pax.pure`."
            )

    def _update_name_to_kind_dict(self, name: str, value):
        """Update the `name_to_kind` dictionary.

        Create a new dictionary and wrap it with `MappingProxyType` to avoid side effects."""
        self._assert_mutability()

        new_dict = OrderedDict(self._pax.name_to_kind)
        new_dict[name] = value
        new_info = self._pax._replace(name_to_kind=MappingProxyType(new_dict))
        super().__setattr__("_pax", new_info)

    def __setattr__(self, name: str, value: Any) -> None:
        """Whenever a user sets an attribute, we will check the assignment:

        - Setting `_pax` is forbidden.
        - If `value` is a pytree of modules and `name` is not in `name_to_kind`,
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
                except TypeError:
                    new_leaves.append(leaf)

            aux = jax.tree_unflatten(treedef, leaves)

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        module_dict = module.__dict__
        module_dict.update(aux)
        # don't have to copy `name_to_kind` anymore, speed thing up!
        # md["name_to_kind"] = OrderedDict(module.name_to_kind)
        module_dict.update(zip(module._pax.name_to_kind, children))

        # if a module is created inside a `pure` function, it is mutable.
        if STATE.inside_pure_function:
            STATE.mutable_module_id_list = (id(module),) + STATE.mutable_module_id_list

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

        def all_module_leaves(tree):
            leaves = jax.tree_flatten(
                tree, is_leaf=lambda m: isinstance(m, BaseModule)
            )[0]
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

    def apply(self: T, apply_fn) -> T:
        """Apply a function to all submodules.

        **Note**: this function returns a transformed copy of the module.

        Arguments:
            apply_fn: a function which inputs a module and outputs a transformed module.
            check_treedef: check treedef before applying the function.
        """

        def rec_fn(mod_or_ndarray):
            if isinstance(mod_or_ndarray, BaseModule):
                return mod_or_ndarray.apply(apply_fn)
            else:
                return mod_or_ndarray

        submodules = self.submodules()
        new_self = jax.tree_map(
            rec_fn,
            self,
            is_leaf=lambda x: isinstance(x, BaseModule) and (x in submodules),
        )

        # tree_map already created a copy of self,
        # hence `apply_fn` is guaranteed to have no side effects.
        return apply_fn(new_self)

    def submodules(self) -> List[T]:
        """Return a list of submodules."""
        module_subtrees = [
            getattr(self, name)
            for name, kind in self._pax.name_to_kind.items()
            if kind == PaxFieldKind.MODULE
        ]

        is_module = lambda x: isinstance(x, BaseModule)
        submods, _ = jax.tree_flatten(module_subtrees, is_leaf=is_module)
        return [v for v in submods if is_module(v)]

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        """This is the default summary method.

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

        def indent(lines: List[str], start_string) -> List[str]:
            return [start_string + l for l in lines]

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

    def _repr(self, info: Optional[Dict[str, Any]] = None) -> str:
        name = f"({self._pax.name}) " if self._pax.name is not None else ""
        cls_name = self.__class__.__name__
        if info is None:
            return f"{name}{cls_name}"
        else:
            lst_info = [f"{k}={v}" for (k, v) in info.items() if v is not None]
            str_info = ", ".join(lst_info)
            return f"{name}{cls_name}[{str_info}]"

    def __repr__(self) -> str:
        return self._repr()
