"""
Manage the global variable ``state``.
"""

import threading
from typing import Any

state = threading.local()
state._rng_key = None
state._seed = None
state.enable_deep_copy = False
state.inside_pure_function = False
state.mutable_module_list = ()


class enable_deep_copy(object):
    r"""A context manager that turns on deepcopy mode."""

    def __init__(self):
        super().__init__()
        self.prev = state.enable_deep_copy

    def __enter__(self):
        self.prev = state.enable_deep_copy
        state.enable_deep_copy = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state.enable_deep_copy = self.prev


class allow_mutation(object):
    r"""A context manager that turns on mutability."""

    def __init__(self, modules):
        from .module import Module

        super().__init__()
        if isinstance(modules, Module):
            modules = (modules,)
        self.mods = tuple(modules)

    def __enter__(self):
        self.prev = state.mutable_module_list
        state.mutable_module_list = self.mods + state.mutable_module_list
        self.prev_inside = state.inside_pure_function
        state.inside_pure_function = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state.mutable_module_list = self.prev
        state.inside_pure_function = self.prev_inside
