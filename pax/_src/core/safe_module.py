"""Safeguards to prevent potential bugs."""

from typing import Iterable, List, Type, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseModule
from .threading_local import allow_mutation, is_mutable

T = TypeVar("T")


class SafeBaseModuleMetaclass(type):
    """Metaclass for `SafeBaseModule`."""

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


class SafeBaseModule(BaseModule, metaclass=SafeBaseModuleMetaclass):
    """Adding safe guards to BaseModule to prevent bugs."""

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
                            f"\n"
                            f"Field `{self}.{name}`:\n"
                            f"    value={value}\n"
                            f"contains a non-module leaf:\n"
                            f"    value={mod}\n"
                            f"    type={type(mod)}\n"
                        )

            # Check if a pytree attribute contains non-ndarray values.
            if name in self._pytree_attributes:
                for leaf in leaves:
                    if not isinstance(leaf, (np.ndarray, jnp.ndarray)):
                        raise ValueError(
                            f"Field `{self}.{name}` contains a non-ndarray value "
                            f"(type={type(leaf)}, value={leaf})."
                        )


def _find_shared_module(module: BaseModule):
    """Find shared module.

    - Return the first module that is shared by two nodes of the pytree.
    - Return `None` if there is no shared module.
    """

    def _get_all_modules(mod: BaseModule, lst: List):
        lst.append(mod)
        is_mod = lambda x: isinstance(x, BaseModule) and x is not mod
        submodules, _ = jax.tree_flatten(mod, is_leaf=is_mod)
        submodules = (m for m in submodules if is_mod(m))
        for m in submodules:
            _get_all_modules(m, lst)

    mods = []
    _get_all_modules(module, mods)
    module_ids = set()
    for m in mods:
        if id(m) in module_ids:
            return m
        module_ids.add(id(m))

    return None
