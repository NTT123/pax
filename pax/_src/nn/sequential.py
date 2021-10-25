"""Sequential module."""

from typing import Optional, Tuple, TypeVar

from ..core import Module
from .lambda_module import Lambda

T = TypeVar("T", bound=Module)


class Sequential(Module):
    """Execute layers in order.

    Support pax.Module (callable pytree) and any jax functions.

    For example:

        >>> net = pax.nn.Sequential(
        ...              pax.nn.Linear(2, 32),
        ...              jax.nn.relu,
        ...              pax.nn.Linear(32, 2)
        ... )
    """

    # Note: we cannot mix pax.Module and jax functions (e.g., jax.nn.relu) in the same list.
    # therefore, we have to convert a jax function to ``Lambda`` module first.
    modules: Tuple[Module, ...]

    def __init__(self, *layers, name: Optional[str] = None):
        """Create a Sequential module."""
        super().__init__(name=name)
        self.modules = tuple(
            (f if isinstance(f, Module) else Lambda(f)) for f in layers
        )

    def __call__(self, x):
        """Call layers in order."""
        for f in self.modules:
            x = f(x)
        return x

    def __getitem__(self, index: int) -> T:
        """Get an item from the `modules` list."""
        return self.modules[index]

    def set(self: T, index: int, value) -> T:
        """Set an item to the `modules` list."""
        if not isinstance(value, Module):
            value = Lambda(value)

        modules = list(self.modules)
        modules[index] = value
        return super().replace(modules=tuple(modules))
