"""
Manage the global variable ``state``.
"""

import threading
from typing import Any

state = threading.local()
state._rng_key = None
state._seed = None
state._enable_deepcopy_wo_treedef = False


class enable_deepcopy_wo_treedef(object):
    r"""A context manager that turns on deepcopy mode."""

    def __init__(self):
        super().__init__()
        self.prev = state._enable_deepcopy_wo_treedef

    def __enter__(self):
        self.prev = state._enable_deepcopy_wo_treedef
        state._enable_deepcopy_wo_treedef = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state._enable_deepcopy_wo_treedef = self.prev


class mutable(object):
    r"""A context manager that turns on mutable mode."""

    def __init__(self, mod):
        super().__init__()
        self.mod = mod
        self.prev = mod._pax.frozen

    def __enter__(self):
        self.prev = self.mod._pax.frozen
        self.mod.__dict__["_pax"] = self.mod._pax._replace(frozen=False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.mod.__dict__["_pax"] = self.mod._pax._replace(frozen=self.prev)
