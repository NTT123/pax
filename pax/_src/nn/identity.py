from ..module import Module


class Identity(Module):
    """Identity function as a module."""

    def __init__(self, *, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        """return x"""
        return x
