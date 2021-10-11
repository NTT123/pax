"""PAX module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

import jax
import jax.tree_util

from .ctx import allow_mutation
from .base import BaseModule, PaxFieldKind
from .transforms import (
    enable_train_mode,
    enable_eval_mode,
    select_parameters,
    update_parameters,
)

T = TypeVar("T", bound="Module")
M = TypeVar("M")
TreeDef = Any


class Module(BaseModule):
    @property
    def training(self) -> bool:
        """If a module is in training mode."""
        return self._pax.training

    @property
    def name(self) -> Optional[str]:
        """Return the name of the module."""
        return self._pax.name

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
        assign its kind to ``PaxFieldKind.MODULE`` in the ``name_to_kind`` dictionary."""

        if hasattr(self, name):
            raise RuntimeError("Cannot register an existing attribute")

        self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)
        setattr(self, name, value)

    register_parameters = register_parameter
    register_states = register_state
    register_module = register_modules

    def copy(self: T) -> T:
        """Return a copy of current module."""
        leaves, treedef = jax.tree_flatten(self)
        return jax.tree_unflatten(treedef, leaves)

    def submodules(self) -> List["Module"]:
        """Return a list of submodules."""
        module_subtrees = [
            getattr(self, name)
            for name, kind in self._pax.name_to_kind.items()
            if kind == PaxFieldKind.MODULE
        ]

        is_module = lambda x: isinstance(x, Module)
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

    def __repr__(self, info: Optional[Dict[str, Any]] = None) -> str:
        name = f"({self._pax.name}) " if self._pax.name is not None else ""
        cls_name = self.__class__.__name__
        if info is None:
            return f"{name}{cls_name}"
        else:
            lst_info = [f"{k}={v}" for (k, v) in info.items() if v is not None]
            str_info = ", ".join(lst_info)
            return f"{name}{cls_name}[{str_info}]"

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

    def train(self: T) -> T:
        """Return a module in training mode."""
        return enable_train_mode(self)

    def eval(self: T) -> T:
        """Return a module in evaluation mode."""
        return enable_eval_mode(self)

    def parameters(self: T) -> T:
        """Return trainable parameters."""
        return select_parameters(self)

    def update_parameters(self: T, params: T) -> T:
        """Return a new module with updated parameters."""
        return update_parameters(self, params=params)

    def find_and_register_submodules(self):
        """Find unregistered submodules and register it with MODULE kind."""

        def all_module_leaves(x):
            leaves = jax.tree_flatten(x, is_leaf=lambda m: isinstance(m, Module))[0]
            return len(leaves) > 0 and all(isinstance(m, Module) for m in leaves)

        for name, value in vars(self).items():
            if name not in self._pax.name_to_kind:
                if all_module_leaves(value):
                    self._update_name_to_kind_dict(name, PaxFieldKind.MODULE)

    def replace(self: T, **kwargs) -> T:
        """Return a new module with some attributes replaced."""

        mod = self.copy()
        with allow_mutation(mod):
            for name, value in kwargs.items():
                assert hasattr(mod, name)
                setattr(mod, name, value)
            mod.find_and_register_submodules()

        mod.scan_bugs()
        return mod

    def apply(self: T, apply_fn) -> T:
        """Apply a function to all submodules.

        **Note**: this function returns a transformed copy of the module.

        Arguments:
            apply_fn: a function which inputs a module and outputs a transformed module.
            check_treedef: check treedef before applying the function.
        """

        def rec_fn(mod_or_ndarray):
            if isinstance(mod_or_ndarray, Module):
                return mod_or_ndarray.apply(apply_fn)
            else:
                return mod_or_ndarray

        submodules = self.submodules()
        new_self = jax.tree_map(
            rec_fn,
            self,
            is_leaf=lambda x: isinstance(x, Module) and (x in submodules),
        )

        # tree_map already created a copy of self,
        # hence `apply_fn` is guaranteed to have no side effects.
        return apply_fn(new_self)

    def scan_bugs(self: T) -> T:
        """Scan the module for potential bugs."""

        def _scan_apply_fn(mod: T) -> T:
            assert isinstance(mod, Module)
            mod._scan_fields(mod.__class__.__dict__)
            mod._scan_fields(mod.__dict__)
            return mod

        self.apply(_scan_apply_fn)
        return self
