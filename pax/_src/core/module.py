"""PAX module."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util

from .base import BaseModule, EmptyNode
from .module_and_value import module_and_value
from .threading_local import allow_mutation

T = TypeVar("T", bound="Module")
M = TypeVar("M")
TreeDef = Any


def parameters_method(trainable_attributes: Iterable[str], submodules=True):
    """Return a `parameters()` method."""

    def _parameters_method(self: T) -> T:
        mod = self.apply_submodules(lambda x: x.parameters())
        for name in mod.pytree_attributes:
            value = getattr(mod, name)
            if submodules:
                leaves, _ = jax.tree_flatten(
                    value, is_leaf=lambda x: isinstance(x, BaseModule)
                )
                has_submod = any(mod for mod in leaves if isinstance(mod, BaseModule))
                if not (has_submod or name in trainable_attributes):
                    mod = mod.replace(**{name: EmptyNode()})
            else:
                if name not in trainable_attributes:
                    mod = mod.replace(**{name: EmptyNode()})
        return mod

    return _parameters_method


def update_pytree(mod: T, *, other: T) -> T:
    """Use non-EmptyNode leaves from other."""

    def _select_fn(leaf_x, leaf_y):
        if isinstance(leaf_y, EmptyNode):
            return leaf_x
        else:
            return leaf_y

    is_empty = lambda x: isinstance(x, EmptyNode)
    new_mod = jax.tree_map(_select_fn, mod, other, is_leaf=is_empty)
    return new_mod


class Module(BaseModule):
    """The base class for all PAX modules.

    Example:

    >>> class Counter(pax.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.count = jnp.array(0)
    ...
    ...     def step(self, x):
    ...         self.count += 1
    """

    _name: Optional[str] = None

    def __init__(self, name: Optional[str] = None):
        """Initialize module.

        >>> linear = pax.nn.Linear(3, 3, name="input_layer")
        >>> print(linear)
        (input_layer) Linear(in_dim=3, out_dim=3, with_bias=True)
        """
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def training(self) -> bool:
        """Return `True` if a module is in training mode.

        >>> net = pax.nn.Linear(1, 1)
        >>> net.training
        True
        >>> net = net.eval()
        >>> net.training
        False
        """
        return self._training

    def apply_submodules(self: M, func: Callable[..., M]) -> M:
        """Apply a function to all submodules, recursively."""
        module = self.copy()
        submod_fn = lambda x: isinstance(x, Module) and x is not module
        leaves, treedef = jax.tree_flatten(module, is_leaf=submod_fn)
        new_leaves = []
        for value in leaves:
            if isinstance(value, Module):
                new_leaves.append(value.apply(func))
            else:
                new_leaves.append(value)
        return jax.tree_unflatten(treedef, new_leaves)

    register_parameter = lambda self, n, v: setattr(self, n, v)
    register_state = register_parameter
    register_modules = register_parameter
    register_parameters = register_parameter
    register_states = register_parameter
    register_module = register_parameter

    parameters = parameters_method((), submodules=True)

    def copy(self: T) -> T:
        """Return a copy of the current module."""
        leaves, treedef = jax.tree_flatten(self)
        return jax.tree_unflatten(treedef, leaves)

    def train(self: T) -> T:
        """Return a module in training mode."""
        return self.apply(lambda mod: mod.replace(_training=True))

    def eval(self: T) -> T:
        """Return a module in evaluation mode."""
        return self.apply(lambda mod: mod.replace(_training=False))

    def update_parameters(self: T, params: T) -> T:
        """Return a new module with updated parameters."""
        return update_pytree(self, other=params.parameters())

    def replace_method(self: T, **methods) -> T:
        cls = self.__class__
        cls_name = cls.__name__
        cls = type(cls_name, (cls,), methods)
        obj = object.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

    def replace(self: T, **kwargs) -> T:
        """Return a new module with some attributes replaced.

        >>> net = pax.nn.Linear(2, 2)
        >>> net = net.replace(bias=jnp.zeros((2,)))
        """

        mod = self.copy()
        with allow_mutation(mod):
            for name, value in kwargs.items():
                assert hasattr(mod, name)
                setattr(mod, name, value)
            mod.find_and_register_pytree_attributes()

        mod.scan_bugs()
        return mod

    # inspired by patrick-kidger/equinox `tree_at`
    def replace_node(self: T, node: jnp.ndarray, value: jnp.ndarray) -> T:
        """Replace a node of the pytree by a new value.

        Example:

        >>> mod = pax.nn.Sequential(
        ...     pax.nn.Linear(2,2),
        ...     jax.nn.relu
        ... )
        >>> mod = mod.replace_node(mod[0].weight, jnp.zeros((2, 3)))
        >>> print(mod[0].weight.shape)
        (2, 3)
        """
        leaves, tree_def = jax.tree_flatten(self, is_leaf=lambda x: x is node)
        count = sum(1 if x is node else 0 for x in leaves)

        if count != 1:
            raise ValueError(f"The node `{node}` appears {count} times in the module.")

        # replace `node` by value
        new_leaves = [value if v is node else v for v in leaves]
        mod: T = jax.tree_unflatten(tree_def, new_leaves)
        mod.scan_bugs()
        return mod

    def scan_bugs(self: T) -> T:
        """Scan the module for potential bugs."""

        # scan for shared module/weight.
        self._assert_not_shared_module()
        self._assert_not_shared_weight()

        def _scan_field_fn(mod: T) -> T:
            assert isinstance(mod, Module)
            # pylint: disable=protected-access
            mod._scan_fields(mod._class_fields())
            # pylint: disable=protected-access
            mod._scan_fields(mod.__dict__.keys())
            return mod

        self.apply(_scan_field_fn)
        return self

    def __mod__(self: T, args: Union[Any, Tuple]) -> Tuple[T, Any]:
        """An alternative to `pax.module_and_value`.

        >>> bn = pax.nn.BatchNorm1D(3)
        >>> x = jnp.ones((5, 8, 3))
        >>> bn, y = bn % x
        >>> bn
        BatchNorm1D(num_channels=3, ...)
        """
        assert callable(self)

        if isinstance(args, tuple):
            return module_and_value(self)(*args)
        else:
            return module_and_value(self)(args)

    def __or__(self: T, other: T) -> T:
        """Merge two modules.

        >>> a = pax.nn.Linear(2, 2)
        >>> b = pax.nn.Linear(2, 2)
        >>> c = a | b
        >>> d = b | a
        >>> c == d
        False
        """
        return update_pytree(self, other=other)

    def __invert__(self: T) -> T:
        return self.parameters()

    def map(self: T, func, *mods) -> T:
        return jax.tree_map(func, self, *mods)

    def _repr(self, info: Optional[Dict[str, Any]] = None) -> str:
        name = f"({self.name}) " if self.name is not None else ""
        cls_name = self.__class__.__qualname__
        if info is None:
            return f"{name}{cls_name}"
        else:
            lst_info = [f"{k}={v}" for (k, v) in info.items() if v is not None]
            str_info = ", ".join(lst_info)
            return f"{name}{cls_name}({str_info})"

    def __repr__(self) -> str:
        return self._repr()

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        """Summarize a module as a tree of its submodules.

        Arguments:
            return_list: return a list of lines instead of a joined string.

        >>> net = pax.nn.Sequential(pax.nn.Linear(2, 3), jax.nn.relu, pax.nn.Linear(3, 4))
        >>> print(net.summary())
        Sequential
        ├── Linear(in_dim=2, out_dim=3, with_bias=True)
        ├── x => relu(x)
        └── Linear(in_dim=3, out_dim=4, with_bias=True)
        """

        output = [self.__repr__()]
        if output[0] is None:
            raise ValueError(
                f"The `{self.__class__}.__repr__` method returns a `None` value."
            )
        submodules: List[Module] = self.submodules()

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

    def set_attribute(self: T, name, value) -> T:
        """Create a new module and set attribute."""
        module = self.copy()
        with allow_mutation(module):
            setattr(module, name, value)
            module.find_and_register_pytree_attributes()

        return module
