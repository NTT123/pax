"""Modules with default kinds"""


from typing import Iterable, Type, TypeVar

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

    def _scan_fields(self, fields: Iterable[str]):
        for field_name in fields:
            if field_name in self._pax.name_to_kind:
                kind = self._pax.name_to_kind[field_name]
                if kind != PaxKind.PARAMETER:
                    raise ValueError(
                        f"`{self.__class__.__name__}.{field_name}` has kind `{kind.name}`. "
                        f"This is not allowed for a ParameterModule object."
                    )
        return super()._scan_fields(fields)


class StateModule(Module):
    """A PAX module that registers STATE by default"""

    def __new__(cls: Type[T], *args, **kwargs) -> T:
        module = super().__new__(cls, *args, **kwargs)
        with allow_mutation(module):
            module._update_default_kind(PaxKind.STATE)
        return module

    def _scan_fields(self, fields: Iterable[str]):
        for field_name in fields:
            if field_name in self._pax.name_to_kind:
                kind = self._pax.name_to_kind[field_name]
                if kind != PaxKind.STATE:
                    raise ValueError(
                        f"`{self.__class__.__name__}.{field_name}` has kind `{kind.name}`. "
                        f"This is not allowed for a StateModule object."
                    )
        return super()._scan_fields(fields)
