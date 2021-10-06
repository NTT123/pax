"""
Manage the global variable ``state``.
"""

import threading
from typing import Any

state = threading.local()
state._rng_key = None
state._seed = None
