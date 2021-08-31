"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", bound="Module")
FilterFn = Callable[[Any, T], bool]

# TODO: use NamedTuple, but, it is slower :-(
ModuleAuxiliaryData = Tuple

# All supported module's field kinds
class PaxFieldKind(Enum):
    STATE: int = 1  # a non-trainable ndarray
    PARAMETER: int = 2  # a trainable ndarray
    MODULE: int = 3  # a Pax Module
    STATE_SUBTREE: int = 4  # a non-trainable pytree
    PARAMETER_SUBTREE: int = 5  # a trainable pytree
    MODULE_SUBTREE: int = 6  # a tree of sub-modules
    OTHERS: int = 7  # all other fields


class ForceModuleInitFakeDict(object):
    """TODO: This is a hack. Fix this!"""

    def __setitem__(self, _, __):
        raise RuntimeError(
            "You may forgot to call `super().__init__()`` "
            "inside your pax.Module's ``__init__`` method."
        )


class Module:
    # Field Name To Kind
    _name_to_kind: Dict[str, PaxFieldKind] = ForceModuleInitFakeDict()
    _training: bool = True

    def __init__(self):
        super().__init__()
        self.__dict__["_name_to_kind"] = dict()
        self.__dict__["_training"] = True

    @property
    def training(self) -> bool:
        return self._training

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_name_to_kind", "_training"]:
            raise RuntimeError(
                f"You SHOULD NOT modify `{name}`. "
                f"If you _really_ want to, use `self.__dict__[name] = value` instead."
            )
        self.__dict__[name] = value
        if isinstance(value, Module):
            self.register_module(name, value)

    def register_parameter(self, name: str, value: jnp.ndarray):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.PARAMETER

    def register_state(self, name: str, value: jnp.ndarray):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.STATE

    def register_module(self, name: str, value: Any):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.MODULE

    def register_parameter_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.PARAMETER_SUBTREE

    def register_state_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.STATE_SUBTREE

    def register_module_subtree(self, name: str, value: Any):
        self.__dict__[name] = value
        self._name_to_kind[name] = PaxFieldKind.MODULE_SUBTREE

    def tree_flatten(self):
        fields = vars(self)

        _tree = {}
        _not_tree = {}
        name_to_kind = self._name_to_kind

        for name, value in fields.items():
            (_tree if name in name_to_kind else _not_tree)[name] = value

        return _tree.values(), (_tree.keys(), _not_tree)

    @classmethod
    def tree_unflatten(cls, aux_data: ModuleAuxiliaryData, children):
        module = cls.__new__(cls)
        _tree, _not_tree = aux_data
        md = module.__dict__
        md.update(_not_tree)
        md["_name_to_kind"] = dict(module._name_to_kind)
        md.update(zip(_tree, children))

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def filter(self: T, keep: str = "parameter") -> T:
        """Filtering a module"""
        assert keep in ["parameter", "state"]
        fields = vars(self)
        cls = self.__class__
        module = cls.__new__(cls)

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

    def update(self: T, other: T) -> T:
        return jax.tree_map(lambda s, o: (s if o is None else o), self, other)

    def train(self: T, mode=True):
        """Rebuild a new model recursively and set `self._training = mode`."""
        submods, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        new_submods = []
        for mod in submods:
            if isinstance(mod, Module):
                new_submods.append(mod.train(mode=mode))
            else:
                new_submods.append(mod)
        model = jax.tree_unflatten(treedef, new_submods)
        model.__dict__["_training"] = mode
        return model

    def eval(self: T) -> T:
        return self.train(False)

    def parameters(self):
        """Return trainable parameters of the module.

        Apply `self.param_filter_fn` if available."""
        params = self.filter("parameter")
        return params

    def freeze(self: T) -> T:
        """Convert all trainable parameters to non-trainable states."""
        submods, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        new_submods = []
        for mod in submods:
            if isinstance(mod, Module):
                new_submods.append(mod.freeze())
            else:
                new_submods.append(mod)
        model = jax.tree_unflatten(treedef, new_submods)
        model.__dict__["_name_to_kind"] = dict(
            model._name_to_kind
        )  # copy to avoid side effects.
        name_to_kind = model._name_to_kind
        for k, v in name_to_kind.items():
            if v == PaxFieldKind.PARAMETER:
                name_to_kind[k] = PaxFieldKind.STATE
            elif v == PaxFieldKind.PARAMETER_SUBTREE:
                name_to_kind[k] = PaxFieldKind.STATE_SUBTREE
        return model

    def hk_init(self, *args, enable_jit=False, **kwargs):
        """Return a new initialized module.

        Arguments:
        enable_jit: bool, if using `jax.jit` for the init function.
        """

        def init_fn(mod, args, kwargs):
            mod = mod.copy()
            mod(*args, **kwargs)
            return mod

        if enable_jit:
            init_fn = jax.jit(init_fn)
        return init_fn(self, args, kwargs)

    def sub_modules(self):
        """Return a list of sub-modules."""
        submods, _ = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
        return [module for module in submods if isinstance(module, Module)]

    def __repr__(self) -> str:
        return self.__class__.__name__

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        """This is the default summary method.
        A module can customize its summary by overriding this function.

        Arguments:
            - return_list: bool, return a list of lines instead of a joined string.

        Expected output:
        Sequential
            Linear[32, 43]
            Linear[5, 5]
        """

        output = [self.__repr__()]
        sub_modules = self.sub_modules()

        def indent(lines: List[str], s) -> List[str]:
            return [s + l for l in lines]

        for i, module in enumerate(sub_modules):
            lines = module.summary(return_list=True)
            if i + 1 < len(sub_modules):  # middle submodules
                _lines = indent(lines[:1], "├── ") + indent(lines[1:], "│   ")
            else:  # last submodule
                _lines = indent(lines[:1], "└── ") + indent(lines[1:], "    ")
            output.extend(_lines)
        if return_list:
            return output
        else:
            return "\n".join(output)
