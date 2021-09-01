from typing import Callable, List, Optional, Union

import jax

from ..module import Module
from ..utils import Lambda


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
        self.register_module_subtree(
            "modules", [(f if isinstance(f, Module) else Lambda(f)) for f in layers]
        )

    def __call__(self, x):
        for f in self.modules:
            x = f(x)
        return x
