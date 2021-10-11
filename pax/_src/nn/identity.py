"""Identity module."""

from ..core import Module


class Identity(Module):
    """Identity function as a module."""

    def __call__(self, x):
        """return x"""
        return x
