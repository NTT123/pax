from typing import Callable, List, Optional, Union

import jax

from ..module import Module


class Lambda(Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __repr__(self) -> str:
        return f"Fx[{self.f}]"

    def summary(self, return_list: bool = False) -> Union[str, List[str]]:
        if self.f == jax.nn.relu:
            name = "relu"
        else:
            name = f"{self.f}"
        output = f"x => {name}(x)"
        return [output] if return_list else output


class Sequential(Module):
    """Execute layers in order.

    Support pax.Module (callable pytree) and any jax functions. For example:
        net = pax.nn.Sequential(
            pax.nn.Linear(2, 32),
            jax.nn.relu,
            pax.nn.Linear(32, 2)
        )
    """

    # Note: we cannot mix pax.Module and jax functions (e.g., jax.nn.relu) in the same list.
    # therefore, we have to convert a jax function to ``Lambda`` module first.
    modules: List[Optional[Module]]
    functions: List[Optional[Callable]]

    def __init__(self, *layers):
        super().__init__()
        filter_fn = lambda f: isinstance(f, Module)
        self.register_parameter_subtree(
            "modules", [(f if isinstance(f, Module) else Lambda(f)) for f in layers]
        )

    def __call__(self, x):
        for f in self.modules:
            x = f(x)
        return x
