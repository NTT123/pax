"""PAX BaseModule."""

# Note: This file is originated from
# https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
# which is under MIT License.

import functools
from copy import deepcopy
from typing import Any, Iterable, List, Mapping, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

# pylint: disable=no-name-in-module
from jaxlib.xla_extension import CompiledFunction

from .threading_local import allow_mutation, is_deep_copy_enabled, is_mutable

T = TypeVar("T", bound="BaseModule")
M = TypeVar("M")


class BaseModuleMetaclass(type):
    """Metaclass for `BaseModule`."""

    def __call__(cls: Type[T], *args, **kwargs) -> T:
        module = cls.__new__(cls, *args, **kwargs)  # type: ignore

        with allow_mutation(module):
            cls.__init__(module, *args, **kwargs)
            module.find_and_register_pytree_attributes()

        # scan module after initialization for potential bugs
        if hasattr(module, "__slots__"):
            raise ValueError("`__slots__` is not supported by PAX modules.")
        module._assert_not_shared_module()
        module._assert_not_shared_weight()
        module._scan_fields(module.__dict__.keys())
        return module


class BaseModule(metaclass=BaseModuleMetaclass):
    """BaseModule manages all information related to the pytree.

    There are two important methods:

    - ``tree_flatten`` converts a module to ``(leaves, treedef)``
    - ``tree_unflatten`` restores the module.

    BaseModule maintains a ``_pytree_attributes`` tuple that lists all subtree attribute names.
    """

    _training: bool
    _pytree_attributes: Tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        self._training = True
        self._pytree_attributes = ()

    @property
    def pytree_attributes(self):
        return self._pytree_attributes

    def replace_method(self: T, **methods) -> T:
        cls = self.__class__
        cls_name = cls.__name__
        cls = type(cls_name, (cls,), methods)
        obj = object.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

    def find_and_register_pytree_attributes(self: T):
        """Find and register ndarrays and submodules."""
        self._assert_mutability()

        pytree_attributes = []
        for name, value in self.__dict__.items():
            leaves, _ = jax.tree_flatten(
                value, is_leaf=lambda x: isinstance(x, (BaseModule, EmptyNode))
            )
            is_pytree = lambda x: isinstance(
                x, (jnp.ndarray, np.ndarray, BaseModule, EmptyNode)
            )
            if any(map(is_pytree, leaves)):
                pytree_attributes.append(name)
        super().__setattr__("_pytree_attributes", tuple(pytree_attributes))

    def _class_fields(self):
        for name, value in self.__class__.__dict__.items():
            if not hasattr(value, "__get__"):  # ignore descriptors
                yield name

    def _assert_mutability(self):
        if not is_mutable(self):
            raise ValueError(
                "Cannot modify a module in immutable mode.\n"
                "Please do this computation inside a function decorated by `pax.pure`."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self._assert_mutability()
        super().__setattr__(name, value)
        self.find_and_register_pytree_attributes()

    def __delattr__(self, name: str) -> None:
        self._assert_mutability()
        super().__delattr__(name)
        self.find_and_register_pytree_attributes()

    def tree_flatten(self) -> Tuple[List[jnp.ndarray], Mapping[str, Any]]:
        """Convert a module to ``(children, treedef)``."""
        aux = dict(self.__dict__)
        children = [aux.pop(name) for name in self._pytree_attributes]

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

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = object.__new__(cls)
        module_dict = module.__dict__
        module_dict.update(aux)
        module_dict.update(zip(module._pytree_attributes, children))
        return module

    def __init_subclass__(cls):
        """Any subclass of ``Module`` is also registered as pytree."""
        jax.tree_util.register_pytree_node_class(cls)

    def _scan_fields(self, fields: Iterable[str]):
        """Scan fields for *potential* bugs."""

        for name in fields:
            value = getattr(self, name)
            is_mod = lambda x: isinstance(x, BaseModule)
            mods, _ = jax.tree_flatten(value, is_leaf=is_mod)
            leaves = jax.tree_leaves(value)
            has_mods = any(map(is_mod, mods))

            # Check if a MODULE attribute contains non-module leafs.
            if has_mods:
                for mod in mods:
                    if not isinstance(mod, BaseModule):
                        raise ValueError(
                            f"Field `{self}.{name}` (value={value}) "
                            f"contains a non-module leaf (type={type(mod)}, value={mod})."
                        )

            # Check if a pytree attribute contains non-ndarray values.
            if name in self._pytree_attributes:
                for leaf in leaves:
                    if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            f"Field `{self}.{name}` contains a non-ndarray value "
                            f"(type={type(leaf)}, value={leaf})."
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

    register_subtree = lambda self, name, value, _: setattr(self, name, value)

    def apply(self: T, apply_fn) -> T:
        """Apply a function to all submodules.

        >>> def print_param_count(mod):
        ...     count = sum(jax.tree_leaves(jax.tree_map(jnp.size, mod)))
        ...     print(f"{count}\t{mod}")
        ...     return mod
        ...
        >>> net = pax.nn.Sequential(pax.nn.Linear(1, 1), jax.nn.relu)
        >>> net = net.apply(print_param_count)
        2 Linear(in_dim=1, out_dim=1, with_bias=True)
        0 Lambda(relu)
        2 Sequential

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
        submod_fn = lambda x: isinstance(x, BaseModule) and x is not self
        leaves, _ = jax.tree_flatten(self, is_leaf=submod_fn)
        return [leaf for leaf in leaves if submod_fn(leaf)]

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
