"""PAX module."""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from .base import EmptyNode, ValueNode
from .module_and_value import module_and_value
from .pure import pure
from .safe_module import SafeBaseModule
from .threading_local import allow_mutation

T = TypeVar("T", bound="Module")
M = TypeVar("M")
TreeDef = Any


def parameters_method(*trainable_attributes, submodules=True):
    """Return a `parameters` method.

    >>> class Linear(pax.Module):
    ...     parameters = pax.parameters_method("weight")
    ...     def __init__(self):
    ...         self.weight = jnp.array(1.0)
    >>> fc = Linear()
    >>> fc == fc.parameters()
    True

    Arguments:
        trainable_atributes: a list of trainable attribute names.
        submodules: include submodules if true.

    Returns:
        A method that returns a module with only trainable weights.
    """

    names = []
    for name in trainable_attributes:
        names.extend(name.split())

    def _parameters(self: T) -> T:
        for name in names:
            assert hasattr(self, name), f"Expecting an attribute with name `{name}`."

        is_submodule = lambda x: x is not self and isinstance(x, SafeBaseModule)
        leaves, treedef = jax.tree_flatten(self, is_leaf=is_submodule)
        leaves = (
            leaf.parameters() if submodules and is_submodule(leaf) else EmptyNode()
            for leaf in leaves
        )
        mod = jax.tree_unflatten(treedef, leaves)
        values = {name: getattr(self, name) for name in names}
        mod = mod.replace(**values)
        return mod

    return _parameters


class Module(SafeBaseModule):
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

    _name: Optional[str]
    _training: bool

    def __new__(cls, *args, **kwargs):
        """Creata a new module.

        Add `_name` and `_training` to __dict__ to make sure that
        the module works correctly even if the __init__ method is
        not called.
        """
        del args, kwargs
        obj = super().__new__(cls)
        obj.__dict__["_name"] = None
        obj.__dict__["_training"] = True
        return obj

    def __init__(self, *, training: bool = True, name: Optional[str] = None):
        """Initialize module."""
        super().__init__()
        self._name = name
        self._training = training

    @property
    def name(self):
        return self._name

    @property
    def training(self) -> bool:
        """Return `True` if a module is in training mode.

        >>> net = pax.Linear(1, 1)
        >>> net.training
        True
        >>> net = net.eval()
        >>> net.training
        False
        """
        return self._training

    def parameters(self):
        """Return a new module with trainable weights only."""
        return parameters_method(submodules=True)(self)

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

        >>> net = pax.Linear(2, 2)
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

        >>> mod = pax.Sequential(
        ...     pax.Linear(2,2),
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

    def set_attribute(self: T, name, value) -> T:
        """Create a new module and set attribute."""
        module = self.copy()
        with allow_mutation(module):
            setattr(module, name, value)
            module.find_and_register_pytree_attributes()

        return module

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        """Summarize a module as a tree of its submodules.

        Arguments:
            return_list: return a list of lines instead of a joined string.

        >>> net = pax.Sequential(pax.Linear(2, 3), jax.nn.relu, pax.Linear(3, 4))
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

    def map(self: T, func, *mods) -> T:
        mods = [mods.state_dict() for mods in mods]
        return self.load_state_dict(jax.tree_map(func, self.state_dict(), *mods))

    def apply(self: T, apply_fn) -> T:
        """Apply a function to all submodules.

        >>> def print_param_count(mod):
        ...     count = sum(jax.tree_leaves(jax.tree_map(jnp.size, mod)))
        ...     print(f"{count}\t{mod}")
        ...     return mod
        ...
        >>> net = pax.Sequential(pax.Linear(1, 1), jax.nn.relu)
        >>> net = net.apply(print_param_count)
        2 Linear(in_dim=1, out_dim=1, with_bias=True)
        0 Lambda(relu)
        2 Sequential

        Arguments:
            apply_fn: a function which inputs a module and outputs a transformed module.
        """

        def rec_fn(mod_or_ndarray):
            if isinstance(mod_or_ndarray, Module):
                return mod_or_ndarray.apply(apply_fn)
            else:
                return mod_or_ndarray

        submodules: List[Module] = self.submodules()
        new_self = jax.tree_map(
            rec_fn,
            self,
            is_leaf=lambda x: isinstance(x, Module) and (x in submodules),
        )

        # tree_map already created a copy of self,
        # hence `apply_fn` is guaranteed to have no side effects.
        return apply_fn(new_self)

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

    def submodules(self) -> List[T]:
        """Return a list of submodules."""
        submod_fn = lambda x: isinstance(x, Module) and x is not self
        leaves, _ = jax.tree_flatten(self, is_leaf=submod_fn)
        return [leaf for leaf in leaves if submod_fn(leaf)]

    def scan_bugs(self: T) -> T:
        """Scan the module for potential bugs."""

        # scan for shared module/weight.
        self._assert_not_shared_module()
        self._assert_not_shared_weight()

        def _scan_field_fn(mod: T) -> T:
            assert isinstance(mod, Module)
            # pylint: disable=protected-access
            mod._scan_fields(mod._class_fields())
            return mod

        self.apply(_scan_field_fn)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Return module's state dictionary."""
        return save_weights_to_dict(self)

    def load_state_dict(self: T, state_dict: Dict[str, Any]) -> T:
        """Return a new module from the state dictionary."""
        return load_weights_from_dict(self, state_dict)

    def __setattr__(self, name: str, value: Any) -> None:
        self._assert_mutability()
        super().__setattr__(name, value)
        # prevent infinite loop
        if name not in ("_pytree_attributes", "_mixed_pytree_attributes"):
            self.find_and_register_pytree_attributes()

    def __delattr__(self, name: str) -> None:
        self._assert_mutability()
        if name in self.pytree_attributes:
            raise ValueError(
                "Cannot delete a pytree attribute.\n"
                "This is to avoid potential bugs related to the order of pytree attributes.\n"
            )
        super().__delattr__(name)

    def __mod__(self: T, args: Union[Any, Tuple]) -> Tuple[T, Any]:
        """An alternative to `pax.module_and_value`.

        >>> bn = pax.BatchNorm1D(3)
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

        >>> a = pax.Linear(2, 2)
        >>> b = pax.Linear(2, 2)
        >>> c = a | b
        >>> d = b | a
        >>> c == d
        False
        """
        return update_pytree(self, other=other)

    def __invert__(self: T) -> T:
        return self.parameters()

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


def save_weights_to_dict(module: Module) -> Dict[str, Any]:
    """Save module weights to a dictionary.

    >>> net = pax.Sequential(pax.Linear(1, 2), jax.nn.relu, pax.Linear(2, 3))
    >>> weights = pax.experimental.save_weights_to_dict(net)
    >>> weights
    {'modules': ({'weight': ..., 'bias': ...}, {}, {'weight':..., 'bias': ...})}
    """

    def _dict_or_array(v):
        if isinstance(v, Module):
            return save_weights_to_dict(v)
        elif isinstance(v, (jnp.ndarray, np.ndarray)):
            return v
        else:
            return None

    out = {}
    for name in module.pytree_attributes:
        value = getattr(module, name)
        out[name] = jax.tree_map(
            _dict_or_array,
            value,
            is_leaf=lambda x: isinstance(x, Module),
        )

    return out


@pure
def load_weights_from_dict(module: T, state_dict: Dict[str, Any]) -> T:
    """Load module weights from a dictionary.

    >>> a = pax.Sequential(pax.Linear(1, 2), jax.nn.relu, pax.Linear(2, 3))
    >>> weights = pax.experimental.save_weights_to_dict(a)
    >>> b = pax.Sequential(pax.Linear(1, 2), jax.nn.relu, pax.Linear(2, 3))
    >>> b = pax.experimental.load_weights_from_dict(b, weights)
    >>> assert a == b
    """

    def _module_or_array(m, s):
        if isinstance(m, Module):
            return load_weights_from_dict(m, s)
        elif isinstance(m, (jnp.ndarray, np.ndarray)):
            return s
        elif s is None:
            return m
        else:
            raise ValueError("Impossible")

    out = module
    for name in module.pytree_attributes:
        value = getattr(module, name)
        value = jax.tree_map(
            _module_or_array,
            value,
            state_dict[name],
            is_leaf=lambda x: isinstance(x, Module),
        )
        setattr(out, name, value)

    return out
