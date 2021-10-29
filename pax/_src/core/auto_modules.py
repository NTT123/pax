"""Modules with automations."""


from typing import Callable, Type, TypeVar

from .base import PaxKind, allow_mutation
from .module import Module

T = TypeVar("T", bound=Module)


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


class AutoModule(Module):
    """A module that auto creates a submodule when needed.


    Example:

    >>> @dataclass
    ... class MLP(pax.AutoModule):
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
    >>> mlp, _ = MLP([1, 2, 3, 4, 5]) % jnp.ones((1, 1))
    >>> print(mlp.summary())
    MLP(features=[1, 2, 3, 4, 5])
    ├── Linear[in_dim=1, out_dim=2, with_bias=True]
    ├── Linear[in_dim=2, out_dim=3, with_bias=True]
    ├── Linear[in_dim=3, out_dim=4, with_bias=True]
    └── Linear[in_dim=4, out_dim=5, with_bias=True]
    """

    def get_or_create(self, name, create_fn: Callable[[], Module]):
        """Create and register a new module when it is not an attribute of the module."""
        if hasattr(self, name):
            module = getattr(self, name)
        else:
            assert callable(create_fn), "Expect a callable function"
            module = create_fn()
            self.register_module(name, module)
        return module
