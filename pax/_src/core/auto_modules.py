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
    """A module that auto creates a submodule when needed."""

    def get_or_create(self, name, create_fn: Callable[[], Module]):
        """Create and register a new module when it is not an attribute of the module."""
        if hasattr(self, name):
            module = getattr(self, name)
        else:
            assert callable(create_fn), "Expect a callable function"
            module = create_fn()
            self.register_module(name, module)
        return module
