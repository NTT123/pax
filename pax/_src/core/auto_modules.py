"""Modules with default kinds"""


from typing import Type, TypeVar

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
