import jax
import jmp

from ..module import Module


class MixedPrecisionModule(Module):
    """Convert the module to a MixedPrecision module."""

    _module: Module

    def __init__(self, mod: Module, mp_policy: jmp.Policy):
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

        if not hasattr(mod, "__call__"):
            raise ValueError("Expecting a callable module.")

    def unwrap_mixed_precision(self):
        """Recreate the original module.

        **Note**: No guarantee that the parameter/state's dtype will be the same as the original module.
        """
        return self._module.copy()

    def __call__(self, *args, **kwargs):
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
        casted_mod, casted_args, casted_kwargs = self.mp_policy.cast_to_compute(
            (self._module, args, kwargs)
        )

        self._module.update(casted_mod, in_place=True)

        casted_mod_clone = self._module.copy()
        # task 2
        output = self._module(*casted_args, **casted_kwargs)

        # task 3
        if jax.tree_structure(self._module) != jax.tree_structure(old_mod_clone):
            raise RuntimeError(
                f"The module `{self._module.__class__.__name__}` has its treedef modified during the forward pass. "
                f"This is currently not supported for a mixed-precision module!"
            )

        def reuse_params_fn(updated_new, new, old):
            # reuse the original parameter if it is
            # NOT modified during the forward pass.
            if updated_new is new:
                return old  # nothing change
            else:
                return self.mp_policy.cast_to_param(updated_new)

        casted_to_param_mod = jax.tree_map(
            reuse_params_fn, self._module, casted_mod_clone, old_mod_clone
        )

        self._module.update(casted_to_param_mod, in_place=True)

        # task 4
        output = self.mp_policy.cast_to_output(output)
        return output
