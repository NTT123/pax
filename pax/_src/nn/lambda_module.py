from typing import Callable, List, Optional, Union

import jax

from ..module import Module


class Lambda(Module):
    """Convert a function to a module."""

    def __init__(self, f: Callable, name: Optional[str] = None):
        super().__init__(name=name)
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __repr__(self, info=None) -> str:
        if self.name is not None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}[{self.f}]"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.name is not None:
            name = self.name
        elif self.f == jax.nn.relu:
            name = "relu"
        else:
            name = f"{self.f}"
        output = f"x => {name}(x)"
        return [output] if return_list else output
