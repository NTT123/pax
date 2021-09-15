"""
Manage the global variable ``state``.
"""

import threading
from typing import Any

state = threading.local()
state._rng_key = None
state._seed = None
state._enable_mutability = True


class mutable(object):
    r"""Context-manager that enables module mutability.

    Example::

        >>> x = pax.nn.Linear(2,2)
        >>> with pax.ctx.mutable():
        ...   x.new_field = 123
        >>> x.new_field
        123
    """

    def __init__(self):
        super().__init__()
        self.prev = state._enable_mutability

    def __enter__(self):
        self.prev = state._enable_mutability
        state._enable_mutability = True

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state._enable_mutability = self.prev


class immutable(object):
    r"""Context-manager that disables module mutability.

    Example::

        >>> x = pax.nn.Linear(2,2)
        >>> with pax.ctx.mutable():
        ...   x.new_field = 123
        >>> x.new_field
        123
    """

    def __init__(self):
        super().__init__()
        self.prev = state._enable_mutability

    def __enter__(self):
        self.prev = state._enable_mutability
        state._enable_mutability = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        state._enable_mutability = self.prev
