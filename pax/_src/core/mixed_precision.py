import functools
from typing import TypeVar

import jax.numpy as jnp
import jmp

from .module import Module, PaxKind

T = TypeVar("T", bound=Module)

# source: https://stackoverflow.com/a/21963090
def _find_descriptor(cls, attrname):
    def hasspecialmethod(obj, name):
        return any(name in klass.__dict__ for klass in type(obj).__mro__)

    for klass in cls.__mro__:
        if attrname in klass.__dict__:
            descriptor = klass.__dict__[attrname]
            if not hasspecialmethod(descriptor, "__get__"):
                return None
            return descriptor
    return None


def _wrap_method(func):
    @functools.wraps(func)
    def mp_method_wrapper(self, *args, **kwargs):
        """A mixed-precision method.

        - Convert all weights to compute dtype.
        - Convert all arguments to compute dtype.
        - Call the original method.
        - Convert all weights to param dtype.
        - Convert output to output dtype.
        """
        old_values = {}
        # pylint: disable=protected-access

        # cast weights to compute dtype
        for name, kind in self._pax.name_to_kind.items():
            if kind in [PaxKind.PARAMETER, PaxKind.STATE]:
                value = getattr(self, name)
                casted_value = self.mp_policy.cast_to_compute(value)
                setattr(self, name, casted_value)
                old_values[name] = value

        args, kwargs = self.mp_policy.cast_to_compute((args, kwargs))
        output = func.__get__(self, type(self))(*args, **kwargs)  # type:ignore

        # cast weights to param dtype
        for name, kind in self._pax.name_to_kind.items():
            if kind in [PaxKind.PARAMETER, PaxKind.STATE]:
                value = getattr(self, name)
                if value is not old_values[name]:  # modified
                    casted_value = self.mp_policy.cast_to_param(value)
                else:
                    casted_value = old_values[name]  # avoid casting operation
                setattr(self, name, casted_value)

        # cast output to output dtype
        output = self.mp_policy.cast_to_output(output)
        return output

    return mp_method_wrapper


def _mp_repr(mp_policy):
    dtype_to_name = {
        jnp.bfloat16: "B",
        jnp.float16: "H",
        jnp.float32: "F",
        jnp.float64: "S",
    }

    return (
        dtype_to_name[mp_policy.param_dtype]
        + dtype_to_name[mp_policy.compute_dtype]
        + dtype_to_name[mp_policy.output_dtype]
    )


def apply_mp_policy(module: T, mp_policy: jmp.Policy) -> T:
    """Create a mixed-precision module.

    Create a subclass on the fly to enfore mixed-precision policy.

    >>> import jmp
    >>> mp_policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float16, output_dtype=jnp.float32)
    >>> net = pax.nn.Linear(3, 3)
    >>> net = pax.apply_mp_policy(net, mp_policy)
    >>> print(net.summary())
    FHF_Linear(in_dim=3, out_dim=3, with_bias=True)
    """

    # pylint: disable=protected-access
    cls_name = f"{_mp_repr(mp_policy)}_{module.__class__.__name__}"
    module_methods = dir(Module)
    base = module.__class__

    methods = {}
    for name in dir(base):
        if name != "__call__" and name.startswith("__"):
            continue
        if name == "__call__" or name not in module_methods:
            value = getattr(base, name)
            if callable(value):
                value = _find_descriptor(base, name)
                if value is None:
                    continue
                if isinstance(value, (staticmethod, classmethod)):
                    methods[name] = value
                else:
                    methods[name] = _wrap_method(value)

    cls = type(cls_name, (base,), methods)
    obj = object.__new__(cls)
    object.__setattr__(obj, "_pax", module._pax)
    obj.__dict__.update(module.__dict__)
    obj.__dict__["mp_policy"] = mp_policy
    for name, kind in obj._pax.name_to_kind.items():
        if kind in [PaxKind.PARAMETER, PaxKind.STATE]:
            obj.__dict__[name] = mp_policy.cast_to_param(obj.__dict__[name])

    return obj
