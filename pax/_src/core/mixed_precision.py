"""Mixed precision computation."""

import inspect
from types import FunctionType
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jmp

from .module import Module
from .threading_local import allow_mutation

TreeDef = Any

T = TypeVar("T", bound=Module)
K = TypeVar("K", bound=Module)
O = TypeVar("O", bound=Module)


class apply_mp_policy(Module, Generic[T]):  # pylint: disable=invalid-name
    """Convert a module to a mixed-precision module."""

    _module: T

    def __init__(self, mod: T, *, mp_policy: jmp.Policy):
        """Create a wrapper module to enforce the mixed-precision policy.
        Arguments:
            mod: the module.
            mp_policy: a ``jmp`` mixed precision policy.
        """
        super().__init__()

        if hasattr(mod, "unwrap_mixed_precision"):
            raise ValueError(
                "Enforcing mixed-precision policy on an object twice is not allowed. "
                "Unwrap it with the `unwrap_mixed_precision` method first!"
            )

        self._module = mp_policy.cast_to_param(mod)
        self.mp_policy = mp_policy

    def unwrap_mixed_precision(self) -> T:
        """Recreate the original module.

        **Note**: No guarantee that the parameter/state's
        dtype will be the same as the original module.
        """
        return self._module.copy()

    def __call__(self, *args, **kwargs):
        return self.__getattr__("__call__")(*args, **kwargs)

    def __getattr__(self, name):
        if not hasattr(self._module, name):
            raise AttributeError

        f = getattr(self._module, name)

        if not inspect.ismethod(f):
            raise ValueError(
                f"Accessing a non-method attribute `{name}` "
                f"of a mixed-precision module is not allowed."
            )

        if isinstance(f, FunctionType):
            raise ValueError(
                f"Calling a static method `{name}` "
                f"is not supported for mixed-precision modules."
            )

        f = getattr(self._module.__class__, name)

        if inspect.ismethod(f):
            raise ValueError(
                f"Calling a class method `{name}` "
                f"is not supported for mixed-precision modules."
            )

        def _fn(*args, **kwargs):
            """This method does four tasks:
            * Task 1: It casts all parameters and arguments to the "compute" data type.
            * Task 2: It calls the original module.
            * Task 3: It casts all the parameters back to the "param" data type.
            However, if a parameter is NOT modified during the forward pass,
            the original parameter will be reused to avoid a `cast` operation.
            * Task 4: It casts the output to the "output" data type.
            """
            old_mod_clone = self._module.copy()

            # task 1
            mod, casted_args, casted_kwargs = self.mp_policy.cast_to_compute(
                (self._module, args, kwargs)
            )

            casted_mod_clone = mod.copy()
            # task 2
            output = f(mod, *casted_args, **casted_kwargs)

            # task 3
            if jax.tree_structure(mod) != jax.tree_structure(old_mod_clone):
                raise ValueError(
                    f"The module `{self._module.__class__.__name__}` has "
                    f"its treedef modified during the forward pass. "
                    f"This is currently not supported for a mixed-precision module!"
                )

            def reuse_params_fn(updated_new, new, old):
                # reuse the original parameter if it is
                # NOT modified during the forward pass.
                if updated_new is new:
                    return old  # nothing change
                else:
                    return self.mp_policy.cast_to_param(updated_new)

            mod = jax.tree_map(reuse_params_fn, mod, casted_mod_clone, old_mod_clone)

            # `mod` has the same pytree structure as `self._module`,
            # therefore, this is safe.
            with allow_mutation(self):
                self._module = mod

            # task 4
            output = self.mp_policy.cast_to_output(output)
            return output

        return _fn

    def __repr__(self):
        dtype_to_name = {
            jnp.bfloat16: "bfloat16",
            jnp.float16: "float16",
            jnp.float32: "float32",
            jnp.float64: "float64",
        }

        info = {
            "param_dtype": dtype_to_name[self.mp_policy.param_dtype],
            "compute_dtype": dtype_to_name[self.mp_policy.compute_dtype],
            "output_dtype": dtype_to_name[self.mp_policy.output_dtype],
        }
        return super()._repr(info=info)
