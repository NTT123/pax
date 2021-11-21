"""Sequential module."""

from typing import Optional, Tuple, TypeVar

from ..core import Module
from .lambda_module import Lambda

T = TypeVar("T", bound=Module)


class Sequential(Module):
    """Execute layers in order.

    Support pax.Module (callable pytree) and any jax functions.

    For example:

    >>> net = pax.Sequential(
    ...              pax.Linear(2, 32),
    ...              jax.nn.relu,
    ...              pax.Linear(32, 3)
    ... )
    >>> print(net.summary())
    Sequential
    ├── Linear(in_dim=2, out_dim=32, with_bias=True)
    ├── x => relu(x)
    └── Linear(in_dim=32, out_dim=3, with_bias=True)
    >>> x = jnp.empty((3, 2))
    >>> y = net(x)
    >>> y.shape
    (3, 3)
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

    def __rshift__(self, other: Module):
        return Sequential(*self.modules, other)
