"""Lambda module."""

from typing import Callable, List, Optional, Union

import jax

from ..core import Module


class Lambda(Module):
    """Convert a function to a module."""

    func: Callable

    def __init__(self, func: Callable, name: Optional[str] = None):
        super().__init__(name=name)
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def __repr__(self) -> str:
        if self.name is not None:
            return super().__repr__()
        else:
            return f"{self.__class__.__name__}[{self.func}]"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.name is not None:
            name = self.name
        elif isinstance(self.func, jax.custom_jvp) and hasattr(self.func, "fun"):
            if hasattr(self.func.fun, "__name__"):
                name = self.func.fun.__name__
            else:
                name = f"{self.func.fun}"
        elif hasattr(self.func, "__name__"):
            name = self.func.__name__
        else:
            name = f"{self.func}"
        output = f"x => {name}(x)"
        return [output] if return_list else output
