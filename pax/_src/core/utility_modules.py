"""Utility Modules."""


from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

import jax
import jax.numpy as jnp

from .module import Module, parameters_method

T = TypeVar("T", bound=Module)
O = TypeVar("O")


class ParameterModule(Module):
    """A PAX module that registers attributes as parameters by default."""

    def parameters(self):
        return self.apply_submodules(lambda x: x.parameters())


class StateModule(Module):
    """A PAX module that registers attributes as states by default."""

    parameters = parameters_method()


class LazyModule(Module):
    """A lazy module is a module that only creates submodules when needed.


    Example:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MLP(pax.experimental.LazyModule):
    ...     features: list
    ...
    ...     def __call__(self, x):
    ...         sizes = zip(self.features[:-1], self.features[1:])
    ...         for i, (in_dim, out_dim) in enumerate(sizes):
    ...             fc = self.get_or_create(f"fc_{i}", lambda: pax.Linear(in_dim, out_dim))
    ...             x = jax.nn.relu(fc(x))
    ...         return x
    ...
    ...
    >>> mlp, _ = MLP([1, 2, 3]) % jnp.ones((1, 1))
    >>> print(mlp.summary())
    MLP(features=[1, 2, 3])
    ├── Linear(in_dim=1, out_dim=2, with_bias=True)
    └── Linear(in_dim=2, out_dim=3, with_bias=True)
    """

    def get_or_create(self, name, create_fn: Callable[[], T]) -> T:
        """Create and register a new attribute when it is not exist.

        Return the attribute.
        """
        if hasattr(self, name):
            value = getattr(self, name)
        else:
            assert callable(create_fn), "Expect a callable function"
            value = create_fn()
            setattr(self, name, value)
        return value


class Lambda(Module):
    """Convert a function to a module.

    Example:

    >>> net = pax.Lambda(jax.nn.relu)
    >>> print(net.summary())
    x => relu(x)
    >>> y = net(jnp.array(-1))
    >>> y
    DeviceArray(0, dtype=int32, weak_type=True)
    """

    func: Callable

    def __init__(self, func: Callable, name: Optional[str] = None):
        super().__init__(name=name)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        if self.name is not None:
            return super().__repr__()
        else:
            return f"{self.__class__.__qualname__}({self.func.__name__})"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.name is not None:
            name = self.name
        elif isinstance(self.func, jax.custom_jvp) and hasattr(self.func, "fun"):
            if hasattr(self.func.fun, "__name__"):
                name = self.func.fun.__name__
            else:
                name = f"{self.func.fun}"
        elif hasattr(self.func, "__name__"):
            name = self.func.__name__
        else:
            name = f"{self.func}"
        output = f"x => {name}(x)"
        return [output] if return_list else output


class Flattener(Module):
    """Flatten PAX modules for better performance.

    Example:

    >>> net = pax.Linear(3, 3)
    >>> opt = opax.adam(1e-3)(net.parameters())
    >>> flat_mods = pax.experimental.Flattener(model=net, optimizer=opt)
    >>> net, opt = flat_mods.model, flat_mods.optimizer
    >>> print(net.summary())
    Linear(in_dim=3, out_dim=3, with_bias=True)
    >>> print(opt.summary())
    chain.<locals>.Chain
    ├── scale_by_adam.<locals>.ScaleByAdam
    │   ├── Linear(in_dim=3, out_dim=3, with_bias=True)
    │   └── Linear(in_dim=3, out_dim=3, with_bias=True)
    └── scale.<locals>.Scale
    """

    treedef_dict: Dict[str, Any]
    leaves_dict: Dict[str, Sequence[jnp.ndarray]]

    def __init__(self, **kwargs):
        """Create a new flattener."""
        super().__init__()
        self.treedef_dict = {}
        self.leaves_dict = {}

        for name, value in kwargs.items():
            leaves, treedef = jax.tree_flatten(value)
            self.treedef_dict[name] = treedef
            self.leaves_dict[name] = leaves

    def __getattr__(self, name: str) -> Any:
        if name in self.treedef_dict:
            treedef = self.treedef_dict[name]
            leaves = self.leaves_dict[name]
            value = jax.tree_unflatten(treedef, leaves)
            return value
        else:
            raise AttributeError()

    def update(self: T, **kwargs) -> T:
        """Update the flattener.

        Example:

        >>> net = pax.Linear(3, 3)
        >>> flats = pax.experimental.Flattener(net=net)
        >>> flats = flats.update(net=pax.Linear(4, 4))
        >>> print(flats.net.summary())
        Linear(in_dim=4, out_dim=4, with_bias=True)
        """
        new_self = self.copy()
        for name, value in kwargs.items():
            leaves, treedef = jax.tree_flatten(value)
            new_self.treedef_dict[name] = treedef
            new_self.leaves_dict[name] = leaves
        return new_self

    def parameters(self: T) -> T:
        """Raise an error.

        Need to reconstruct the original module before getting parameters.
        """

        raise ValueError(
            "A flattener only stores ndarray leaves as non-trainable states.\n"
            "Reconstruct the original module before getting parameters."
        )
