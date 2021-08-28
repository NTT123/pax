from typing import Callable, List, Optional

from ..module import Module


class Sequential(Module):
    """Execute layers in order.

    Support tx.Module (callable pytree) and any jax functions. For example:
        net = pax.nn.Sequential(
            pax.nn.Linear(2, 32),
            jax.nn.relu,
            pax.nn.Linear(32, 2)
        )
    """

    # Note: we cannot mix pax.Module and jax functions (e.g., jax.nn.relu) in the same list.
    modules: List[Optional[Module]]
    functions: List[Optional[Callable]]

    def __init__(self, *layers):
        filter_fn = lambda f: isinstance(f, Module)
        self.modules = [(f if filter_fn(f) else None) for f in layers]
        self.functions = [(f if not filter_fn(f) else None) for f in layers]

    def __call__(self, x):
        for f1, f2 in zip(self.modules, self.functions):
            x = f1(x) if f2 is None else f2(x)
        return x
