"""
Manage the global variable ``state``.
"""

import threading
from typing import Any

state = threading.local()
state._rng_key = None
state._seed = None
state._enable_deep_copy = False
state._enable_mutability = False


class enable_deep_copy(object):
    r"""A context manager that turns on deepcopy mode."""

    def __init__(self):
        super().__init__()
        self.prev = state._enable_deep_copy

    def __enter__(self):
        self.prev = state._enable_deep_copy
        state._enable_deep_copy = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state._enable_deep_copy = self.prev


class enable_mutability(object):
    r"""A context manager that turns on mutability."""

    def __init__(self):
        super().__init__()
        self.prev = state._enable_mutability

    def __enter__(self):
        self.prev = state._enable_mutability
        state._enable_mutability = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state._enable_mutability = self.prev
