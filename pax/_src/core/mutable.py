from contextlib import contextmanager

from .module import Module
from .pure import get_all_submodules
from .threading_local import allow_mutation


@contextmanager
def mutable(module: Module):
    """A context manager that allows a copy module to be mutable inside the context.

    >>> net = pax.Linear(1, 2)
    >>> with pax.experimental.mutable(net) as net:
    ...     net.bias = jnp.array(0.)
    >>> assert net.bias.item() == 0.
    """

    copy = module.copy()
    all_submodules = get_all_submodules(copy)

    with allow_mutation(all_submodules):
        try:
            yield copy
        finally:
            copy.find_and_register_pytree_attributes()
            copy.scan_bugs()
