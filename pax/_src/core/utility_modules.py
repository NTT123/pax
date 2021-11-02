"""Utility Modules."""


from typing import Callable, List, Optional, Type, TypeVar, Union

import jax

from .base import PaxKind, allow_mutation
from .module import Module

T = TypeVar("T", bound=Module)
O = TypeVar("O")


class ParameterModule(Module):
    """A PAX module that registers PARAMETER by default"""

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        module = super().__new__(cls, *args, **kwargs)
        with allow_mutation(module):
            module._update_default_kind(PaxKind.PARAMETER)
        return module


class StateModule(Module):
    """A PAX module that registers STATE by default"""

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        module = super().__new__(cls, *args, **kwargs)
        with allow_mutation(module):
            module._update_default_kind(PaxKind.STATE)
        return module


class LazyModule(Module):
    """A module that auto creates a submodule when needed.


    Example:

    >>> @dataclass
    ... class MLP(pax.LazyModule):
    ...     features: Sequence[int]
    ...
    ...     def __call__(self, x):
    ...         sizes = zip(self.features[:-1], self.features[1:])
    ...         for i, (in_dim, out_dim) in enumerate(sizes):
    ...             fc = self.get_or_create(f"fc_{i}", lambda: pax.nn.Linear(in_dim, out_dim))
    ...             x = jax.nn.relu(fc(x))
    ...         return x
    ...
    ...
    >>> mlp, _ = MLP([1, 2, 3]) % jnp.ones((1, 1))
    >>> print(mlp.summary())
    MLP(features=[1, 2, 3])
    ├── Linear[in_dim=1, out_dim=2, with_bias=True]
    └── Linear[in_dim=2, out_dim=3, with_bias=True]
    """

    def get_or_create(self, name, create_fn: Callable[[], T], kind=PaxKind.MODULE) -> T:
        """Create and register a new attribute when it is not exist.

        Return the attribute.
        """
        if hasattr(self, name):
            value = getattr(self, name)
        else:
            assert callable(create_fn), "Expect a callable function"
            value = create_fn()
            self.register_subtree(name, value, kind)
        return value

    def get_or_create_parameter(self, name, create_fn: Callable[[], O]) -> O:
        """Get or create a trainable parameter."""
        return self.get_or_create(name, create_fn=create_fn, kind=PaxKind.PARAMETER)

    def get_or_create_state(self, name, create_fn: Callable[[], O]) -> O:
        """Get or create a non-trainable state."""
        return self.get_or_create(name, create_fn=create_fn, kind=PaxKind.STATE)

    def parameters(self: T) -> T:
        if len(jax.tree_leaves(self)) == 0:
            raise ValueError("An empty lazy module. Please initialize it!")

        return super().parameters()


class Lambda(Module):
    """Convert a function to a module."""

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
            return f"{self.__class__.__name__}[{self.func}]"

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
