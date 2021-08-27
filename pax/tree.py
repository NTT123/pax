from typing import Any, Dict, cast

import jax
import jax.tree_util
import numpy as np


class Leaf:
    """When flattening, annotated variables will be wrapped by the corresponding annotation types. When unflattening, the value will be unwrapped.

    These wrap and unwrap procedures are used to materialize type information in the pytree structure.
    This is not much different from the treex's approach to use `ValueAnnotation` object to store type information.
    The difference is that we make `ValueAnnotation` a Pytree so it can fit in with jax transformations (e.g., jax.jit, jax.grad) nicely.

    Fun fact: Leaf is not _really_ a pytree leaf, but for all of our purposes, it is a leaf.
    """

    value: Any
    info: Dict[str, Any]

    def __init__(self, value: Any, info: Dict[str, Any] = None):
        self.value = value
        self.info = info

    def tree_flatten(self):
        return [self.value], None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return children[0]

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)


class _Parameter(Leaf):
    pass


class _State(Leaf):
    pass


# use cast to trick static analyzers into believing these types
Parameter = cast(np.ndarray, _Parameter)
State = cast(np.ndarray, _State)


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()

    def __repr__(self) -> str:
        return "Nothing"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Nothing)
