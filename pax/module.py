"""Pax module.

Note: This file is originated from 
https://raw.githubusercontent.com/cgarciae/treex/32e4cce5ca0cc991cda8076903853621d0aa4ab9/treex/module.py
which is under MIT License.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util
import jmp

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
    OTHERS: int = 7


class Module:
    """Module is the central object of Pax.

    It manages all information related to the pytree.
    It also includes methods (usually, ``__call__``) that can be executed to compute functions on the pytree.

    The two important methods: ``flatten`` and ``unflatten`` specify how a module can be converted to a ``(leaves, treedef)``,
    and otherwise from ``(treedef, leaves)`` back to a module.

    A module maintains a ``_name_to_kind`` dictionary that tells if an attribute is part of
    the pytree and the kind of the tree part (parameter, state, module, etc.).
    """

    # Field Name To Kind
    _name_to_kind: Dict[str, PaxFieldKind] = None
    _training: bool = True
    name: str = None

    def __init__(self, name: Optional[str] = None):
        """Initialize the ``_training`` flag (the default is ``True``)
        and the **very** important ``_name_to_kind`` dictionary.

        It is required that any subclass of ``Module`` has to call ``super().__init__()`` for initialization.
        We implement a safeguard mechanism to enforce that by checking if ``_name_to_kind`` is ``None`` in the ``__setattr__`` method.
        """
        super().__init__()
        super().__setattr__("_name_to_kind", dict())
        super().__setattr__("_training", True)
        super().__setattr__("name", name)

    @property
    def training(self) -> bool:
        return self._training

    def __setattr__(self, name: str, value: Any) -> None:
        """Whenever a user sets ``value`` to attribute ``name``, we will check the assignment.

        * Setting ``_name_to_kind`` and ``_training`` are forbidden.

        * Setting ``value`` to a wrong kind attribute is also forbidden.

        * If ``value`` is a ``Module``'s instance and ``name`` is not in ``_name_to_kind``, it will be assigned of kind ``PaxFieldKind.MODULE``.
        """
        if self._name_to_kind is None:
            raise RuntimeError(
                "You forgot to call `super().__init__()`` "
                "inside your pax.Module's ``__init__`` method."
            )

        if name in ["_name_to_kind", "_training"]:
            raise RuntimeError(
                f"You SHOULD NOT modify `{name}`. "
                f"If you _really_ want to, use `self.__dict__[name] = value` instead."
            )

        super().__setattr__(name, value)

        if isinstance(value, Module) and name not in self._name_to_kind:
            self._name_to_kind[name] = PaxFieldKind.MODULE

        self._scan_fields(fields={name: value})

    def register_parameter(self, name: str, value: jnp.ndarray):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.PARAMETER
        setattr(self, name, value)

    def register_state(self, name: str, value: jnp.ndarray):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.STATE
        setattr(self, name, value)

    def register_module(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.MODULE
        setattr(self, name, value)

    def register_parameter_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.PARAMETER_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.PARAMETER_SUBTREE
        setattr(self, name, value)

    def register_state_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.STATE_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.STATE_SUBTREE
        setattr(self, name, value)

    def register_module_subtree(self, name: str, value: Any):
        """Register ``value`` as an attribute of the object under the name ``name`` and
        assign its kind to ``PaxFieldKind.MODULE_SUBTREE`` in the ``_name_to_kind`` dictionary."""

        self._name_to_kind[name] = PaxFieldKind.MODULE_SUBTREE
        setattr(self, name, value)

    def tree_flatten(self):
        """Convert a module to ``(children, treedef)``."""
        fields = vars(self)

        _tree = {}
        _not_tree = {}
        name_to_kind = self._name_to_kind

        for name, value in fields.items():
            (_tree if name in name_to_kind else _not_tree)[name] = value

        return _tree.values(), (_tree.keys(), _not_tree)

    @classmethod
    def tree_unflatten(cls, aux_data: ModuleAuxiliaryData, children):
        """Recreate a module from its ``(children, treedef)``."""
        module = cls.__new__(cls)
        _tree, _not_tree = aux_data
        md = module.__dict__
        md.update(_not_tree)
        md["_name_to_kind"] = dict(module._name_to_kind)
        md.update(zip(_tree, children))

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
        """Rebuild a new model recursively and set ``self._training = mode``.

        The default behavior is to create a new module in ``train`` mode.

        Arguments:
            mode: return a copy module in ``train`` mode module if ``True``.
        """
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
        """Return a copy module in ``eval`` mode."""
        return self.train(False)

    def parameters(self):
        """Return trainable parameters of the module."""
        params = self.filter("parameter")
        return params

    def freeze(self: T) -> T:
        """Return a copy module with all trainable parameters are converted to non-trainable states."""
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

    def hk_init(self, *args, enable_jit: bool = False, **kwargs):
        """Return a new initialized module.

        **Note**: This function is only useful if a module is or includes a converted module from the haiku library.

        Arguments:
            args, kwargs: dummy inputs to the module.
            enable_jit: to use `jax.jit` for the init function.
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

        fields = vars(self)
        self._scan_fields(fields)

        for mod in self.sub_modules():
            mod.deep_scan()

    def _scan_fields(self, fields: Sequence[Any]):
        """Scan fields for *potential* bugs."""

        from jax.dtypes import issubdtype as isdt

        for name, value in fields.items():
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

    def mixed_precision(self: T, mp_policy: jmp.Policy, method_name="__call__"):
        """Convert the module to a MixedPrecision module.

        It operates by creating a new clone object that has one method be wrapped to enforce the mixed-precision policy.

        Arguments:
            mp_policy: a ``jmp`` mixed precision policy.
            method_name: name of the method that will be affected.
        """
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
            casted_self, casted_args, casted_kwargs = mp_policy.cast_to_compute(
                (self_, args, kwargs)
            )
            self_.update(casted_self, in_place=True)
            output = getattr(cls, method_name)(self_, *casted_args, **casted_kwargs)
            output = mp_policy.cast_to_output(output)
            return output

        setattr(MixedPrecisionWrapper, method_name, mp_call)
        MixedPrecisionWrapper.__name__ = self.__class__.__name__ + "@MixedPrecision"
        new = MixedPrecisionWrapper.__new__(MixedPrecisionWrapper)
        new.__dict__.update(casted_self.__dict__)
        return new

    def apply(self, apply_fn):
        """Apply a function to all sub-modules.

        **Note**: this function returns a transformed copy of the current object.

        Arguments:
            apply_fn: a function which inputs a module and outputs a new module.
        """

        def rec_fn(x):
            if isinstance(x, Module):
                return x.apply(apply_fn)
            else:
                return x

        new_self = jax.tree_map(
            rec_fn, self, is_leaf=lambda x: isinstance(x, Module) and x is not self
        )
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
