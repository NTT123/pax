"""PAX BaseModule."""

# Note: This file is originated from
# https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
# which is under MIT License.

import functools
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax.dtypes import issubdtype as isdt

# pylint: disable=no-name-in-module
from jaxlib.xla_extension import CompiledFunction

from .threading_local import allow_mutation, is_deep_copy_enabled, is_mutable

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")


@jax.tree_util.register_pytree_node_class
class EmptyNode(Tuple):
    """We use this class to mark deleted nodes.

    Note: this is inspired by treex's `Nothing` class.
    """

    def tree_flatten(self):
        """Flatten empty node."""
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten empty node."""
        del aux, children
        return EmptyNode()


class PaxKind(Enum):
    """``PaxKind`` lists all supported attribute kinds in ``pax.Module``.

    An attribute will be considered as part of the pytree structure
    if its kind is one of ``STATE``, ``PARAMETER``, ``MODULE``.

    * A ``STATE`` attribute is a non-trainable leaf of the pytree.
    * A ``PARAMETER`` attribute is a trainable leaf of the pytree.
    * A ``MODULE`` attribute is a generic subtree.
    """

    STATE: int = 1
    PARAMETER: int = 2
    MODULE: int = 3
    UNKNOWN: int = -1


class PaxModuleInfo(NamedTuple):
    """PAX Internal Data Structure."""

    training: bool
    name_to_kind: Mapping[str, PaxKind]
    default_kind: PaxKind

    def __repr__(self) -> str:
        nodes = ", ".join([f"{k}:{v.name}" for k, v in self.name_to_kind.items()])
        return f"PaxModuleInfo[training={self.training}, nodes={{{nodes}}}]"


class BaseModuleMetaclass(type):
    """Metaclass for `BaseModule`."""

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        module = cls.__new__(cls, *args, **kwargs)  # type: ignore

        with allow_mutation(module):
            with module._default_kind(module._pax.default_kind):
                with module._default_kind(PaxKind.MODULE):
                    cls.__init__(module, *args, **kwargs)

        # scan module after initialization for potential bugs
        module._assert_not_shared_module()
        module._assert_not_shared_weight()
        module._scan_fields(module.__dict__.keys())
        return module


class BaseModule(metaclass=BaseModuleMetaclass):
    """BaseModule manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    BaseModule maintains a ``name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    __slots__ = ("_pax",)

    _pax: PaxModuleInfo

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        """Initialize `_pax` attribute in `__new__` method to avoid
        calling `super().__init__()` in every subclass of Module."""

        del args, kwargs

        obj = object.__new__(cls)
        super(BaseModule, obj).__setattr__(
            "_pax",
            PaxModuleInfo(
                training=True,
                name_to_kind=MappingProxyType(OrderedDict()),
                default_kind=PaxKind.UNKNOWN,
            ),
        )

        if obj.__slots__ != ("_pax",):
            raise ValueError("`__slots__` is not supported by PAX modules.")

        # scan class attributes for unregistered modules and ndarray's.
        obj._scan_fields(obj.__class__.__dict__)

        return obj

    def _assert_mutability(self):
        if not is_mutable(self):
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
        """Setting `_pax` attribute is forbidden."""
        self._assert_mutability()

        if name == "_pax":
            raise ValueError(
                f"{name} is a reserved attribute for PAX internal mechanisms."
            )

        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        del name
        raise ValueError("Cannot delete a module's attribute.")

    def _update_default_kind(self, kind: PaxKind):
        self._assert_mutability()
        new_pax_info = self._pax._replace(default_kind=kind)
        super(BaseModule, self).__setattr__("_pax", new_pax_info)

    @contextmanager
    def _default_kind(self, kind: PaxKind):
        """Search for any new unregistered pytree attribute
        at the end of the context and register it with kind `kind`.
        """
        fields_before = set(vars(self).keys())
        try:
            yield
        finally:
            fields_after = set(vars(self).keys())
            new_fields = fields_after.difference(fields_before)
            self._find_and_register_pytree(kind, fields=new_fields)

    def add_parameters(self):
        """Add new attributes as trainable parameters"""
        return self._default_kind(PaxKind.PARAMETER)

    def add_states(self):
        """Add new attributes as non-trainable states."""
        return self._default_kind(PaxKind.STATE)

    def tree_flatten(
        self,
    ) -> Tuple[List[jnp.ndarray], Tuple[Mapping[str, Any], PaxModuleInfo]]:
        """Convert a module to ``(children, treedef)``."""

        aux = dict(self.__dict__)
        children = [aux.pop(name) for name in self._pax.name_to_kind]

        if is_deep_copy_enabled():
            leaves, treedef = jax.tree_flatten(aux)
            new_leaves = []
            black_list = (jax.custom_jvp, functools.partial, CompiledFunction)
            for leaf in leaves:
                try:
                    if isinstance(leaf, black_list):
                        new_leaves.append(leaf)
                    else:
                        new_leaf = deepcopy(leaf)
                        new_leaves.append(new_leaf)
                except TypeError:
                    new_leaves.append(leaf)
            aux = jax.tree_unflatten(treedef, new_leaves)

        return children, (aux, self._pax)

    @classmethod
    def tree_unflatten(cls, aux_pax, children):
        """Recreate a module from its ``(children, treedef)``."""
        aux, pax_info = aux_pax
        module = object.__new__(cls)
        super(BaseModule, module).__setattr__("_pax", pax_info)
        module_dict = module.__dict__
        module_dict.update(aux)
        # don't have to copy `name_to_kind` anymore, speed thing up!
        # md["name_to_kind"] = OrderedDict(module.name_to_kind)
        # pylint: disable=protected-access
        module_dict.update(zip(module._pax.name_to_kind, children))

        return module

    def __init_subclass__(cls):
        """Any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def _scan_fields(self, fields: Iterable[str]):
        """Scan fields for *potential* bugs."""

        for name in fields:
            value = getattr(self, name)
            kind = self._pax.name_to_kind.get(name, PaxKind.UNKNOWN)
            mods, _ = jax.tree_flatten(
                value, is_leaf=lambda x: isinstance(x, BaseModule)
            )
            leaves = jax.tree_leaves(value)

            # Check if a MODULE attribute contains non-module leafs.
            if kind == PaxKind.MODULE:
                for mod in mods:
                    if not isinstance(mod, BaseModule):
                        raise ValueError(
                            f"Field `{self}.{name}` (kind={kind}, value={value}) "
                            f"contains a non-module leaf (type={type(mod)}, value={mod})."
                        )

            # Check if a pytree attribute contains non-ndarray values.
            if kind != PaxKind.UNKNOWN:
                for leaf in leaves:
                    if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) contains a non-ndarray value "
                            f"(type={type(leaf)}, value={leaf})."
                        )

            # Check if a PARAMETER attribute contains non-differentiable ndarray's.
            if kind == PaxKind.PARAMETER:
                for leaf in leaves:
                    if hasattr(leaf, "dtype") and not (
                        isdt(leaf.dtype, jnp.complexfloating)
                        or isdt(leaf.dtype, jnp.floating)
                    ):
                        raise ValueError(
                            f"Field `{self}.{name}` ({kind}) contains a non-differentiable leaf "
                            f"(type={leaf.dtype}, value={leaf})."
                        )

            if kind == PaxKind.UNKNOWN:
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
                            f"Unregistered field `{self.__class__.__name__}.{name}`"
                            f" ({kind}) contains a ndarray. Consider registering it using"
                            f" `self.add_parameters()` or `self.add_states()` context managers."
                        )

            # Check if an unregistered (or PARAMETER) field contains pax.Module instances
            if kind not in [PaxKind.MODULE, PaxKind.STATE]:
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

    def _find_and_register_pytree(
        self, kind: PaxKind, fields: Optional[Iterable[str]] = None
    ):
        """Find unregistered pytree attributes and register it with kind `kind`."""

        if kind == PaxKind.UNKNOWN:
            return

        if fields is not None:
            items = ((name, getattr(self, name)) for name in fields)
        else:
            items = vars(self).items()

        def is_pytree_of(tree, classes):
            is_leaf_fn = lambda m: isinstance(m, classes)
            leaves = jax.tree_flatten(tree, is_leaf=is_leaf_fn)[0]
            return len(leaves) > 0 and all(is_leaf_fn(m) for m in leaves)

        for name, value in items:
            if name not in self._pax.name_to_kind:
                if (kind == PaxKind.MODULE and is_pytree_of(value, BaseModule)) or (
                    kind in [PaxKind.PARAMETER, PaxKind.STATE]
                    and not _has_module_node(value)
                    and is_pytree_of(value, (np.ndarray, jnp.ndarray))
                ):
                    self._update_name_to_kind_dict(name, kind)

    def register_subtree(self, name: str, value, kind: PaxKind):
        """Assign `value` to attribute `name` and register its kind as `kind`."""
        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        setattr(self, name, value)
        self.set_attribute_kind(**{name: kind})

    def set_attribute_kind(self, **kwargs):
        """Set PAX kind for one or many attributes.

        Example:

        >>> class M(pax.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = jnp.array(0.)
        ...         self.counter = jnp.array(0)
        ...         self.set_attribute_kind(weight=pax.P, counter=pax.S)
        ...
        >>> m = M()
        >>> print(m._pax.name_to_kind)
        OrderedDict([('weight', <PaxKind.PARAMETER: 2>), ('counter', <PaxKind.STATE: 1>)])
        """
        for name, kind in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"Attribute `{name}` does not exist.")
            self._update_name_to_kind_dict(name, kind)

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

        submodules: List[BaseModule] = self.submodules()
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
            if kind == PaxKind.MODULE
        ]

        is_module = lambda x: isinstance(x, BaseModule)
        submods, _ = jax.tree_flatten(module_subtrees, is_leaf=is_module)
        return [v for v in submods if is_module(v)]

    def _assert_not_shared_module(self):
        """Shared module is not allowed."""
        shared_module = _find_shared_module(self)
        if shared_module is not None:
            raise ValueError(
                f"The module `{shared_module}` is shared between two nodes of the pytree.\n"
                f"This is not allowed to prevent potential silence bugs."
            )

    def _assert_not_shared_weight(self):
        """Shared weight is not allowed."""
        leaves = jax.tree_leaves(self)
        leaf_ids = set()
        for leaf in leaves:
            if id(leaf) in leaf_ids:
                raise ValueError(
                    f"Detected a shared ndarray. This is not allowed.\n"
                    f"Shape={leaf.shape}\n"
                    f"Dtype={leaf.dtype}\n"
                    f"Value={leaf}",
                )
            leaf_ids.add(id(leaf))


def _find_shared_module(module: BaseModule):
    """Find shared module.

    - Return the first module that is shared by two nodes of the pytree.
    - Return `None` if there is no shared module.
    """

    def _get_all_modules(mod: BaseModule, lst: List):
        lst.append(mod)
        for m in mod.submodules():
            _get_all_modules(m, lst)

    mods = []
    _get_all_modules(module, mods)
    module_ids = set()
    for m in mods:
        if id(m) in module_ids:
            return m
        module_ids.add(id(m))

    return None


def _has_module_node(value):
    leaves = jax.tree_flatten(value, is_leaf=lambda x: isinstance(x, BaseModule))[0]
    for leaf in leaves:
        if isinstance(leaf, BaseModule):
            return True

    return False
