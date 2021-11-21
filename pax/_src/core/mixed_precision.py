"""Enforce mixed-precision policy."""

import functools
from typing import TypeVar

import jax
import jax.numpy as jnp
import jmp

from .module import Module
from .safe_module import find_descriptor

T = TypeVar("T", bound=Module)


def _wrap_method(func):
    """Wrap a class's method to enforce mixe-precision policy."""

    @functools.wraps(func)
    def mp_method_wrapper(self, *args, **kwargs):
        """A mixed-precision method.

        - Convert all weights to compute dtype.
        - Cast all arguments to compute dtype.
        - Call the original method.
        - Convert all weights to param dtype.
        - Cast output to output dtype.

        We bypass PAX mutability checking to make mixed-precision
        policy transparent from the user's point of view.
        """
        original_values = {}
        casted_original = {}
        # pylint: disable=protected-access

        # convert weights to compute dtype
        for name in self.pytree_attributes:
            value = getattr(self, name)
            if not _has_module(value):
                casted_value = self._pax_mp_policy.cast_to_compute(value)
                self.__dict__[name] = casted_value
                original_values[name] = value
                casted_original[name] = casted_value

        # cast arguments to compute dtype
        args, kwargs = self._pax_mp_policy.cast_to_compute((args, kwargs))
        output = func.__get__(self, type(self))(*args, **kwargs)  # type:ignore

        # convert weights to param dtype
        for name in self.pytree_attributes:
            value = getattr(self, name)
            if not _has_module(value):
                if value is not casted_original[name]:  # modified
                    casted_value = self._pax_mp_policy.cast_to_param(value)
                    setattr(self, name, casted_value)
                else:
                    # avoid casting operation
                    self.__dict__[name] = original_values[name]

        # cast output to output dtype
        output = self._pax_mp_policy.cast_to_output(output)
        return output

    return mp_method_wrapper


def _mp_repr(mp_policy):
    dtype_to_name = {
        jnp.bfloat16: "H",
        jnp.float16: "H",
        jnp.float32: "F",
        jnp.float64: "F",
    }

    return (
        dtype_to_name[mp_policy.param_dtype]
        + dtype_to_name[mp_policy.compute_dtype]
        + dtype_to_name[mp_policy.output_dtype]
    )


def apply_mp_policy(module: T, mp_policy: jmp.Policy) -> T:
    """Create a mixed-precision module.

    Create a subclass on the fly to enforce the mixed-precision policy.

    >>> import jmp
    >>> mp_policy = jmp.get_policy("params=float32,compute=float16,output=float32")
    >>> net = pax.Linear(3, 3)
    >>> net = pax.apply_mp_policy(net, mp_policy)
    >>> print(net.summary())
    Linear(in_dim=3, out_dim=3, with_bias=True, mp_policy=FHF)
    """

    if hasattr(module, "_pax_mp_policy"):
        raise ValueError(
            "Cannot apply multiple mixed-precision policies on an object.\n"
            "Call `pax.unwrap_mp_policy(...)` to remove the policy first."
        )

    # pylint: disable=protected-access
    cls_name = module.__class__.__name__
    module_methods = dir(Module)
    base = module.__class__

    methods = {}
    for name in dir(base):
        if name != "__call__" and name.startswith("__"):
            continue
        if name == "__call__" or name not in module_methods:
            value = getattr(base, name)
            if callable(value):
                value = find_descriptor(base, name)
                if value is None:
                    continue
                if isinstance(value, (staticmethod, classmethod)):
                    methods[name] = value
                else:
                    methods[name] = _wrap_method(value)

    def _repr(self, info=None):
        if info is None:
            info = {}
        info["mp_policy"] = _mp_repr(self._pax_mp_policy)
        return super(base, self)._repr(info)  # type: ignore

    methods["_repr"] = _repr

    cls = type(cls_name, (base,), methods)
    obj = object.__new__(cls)
    obj.__dict__.update(module.__dict__)
    obj.__dict__["_pax_mp_policy"] = mp_policy
    for name in obj.pytree_attributes:
        value = getattr(obj, name)
        if not _has_module(value):
            obj.__dict__[name] = mp_policy.cast_to_param(obj.__dict__[name])
    return obj


def unwrap_mp_policy(module: T) -> T:
    """Unwrap a mixed-precision module to recreate the original module.

    >>> import jmp
    >>> mp_policy = jmp.get_policy("params=float32,compute=float16,output=float32")
    >>> net = pax.Linear(3, 3)
    >>> net = pax.apply_mp_policy(net, mp_policy)
    >>> print(net.summary())
    Linear(in_dim=3, out_dim=3, with_bias=True, mp_policy=FHF)
    >>> net = pax.unwrap_mp_policy(net)
    >>> print(net.summary())
    Linear(in_dim=3, out_dim=3, with_bias=True)
    """
    if not hasattr(module, "_pax_mp_policy"):
        raise ValueError("Expected a mixed-precision module.")

    base = module.__class__.__base__
    original = object.__new__(base)
    original.__dict__.update(module.__dict__)
    del original.__dict__["_pax_mp_policy"]
    return original


def _has_module(mod):
    is_mod = lambda x: x is not mod
    leaves, _ = jax.tree_flatten(mod, is_leaf=is_mod)
    return any(map(is_mod, leaves))
