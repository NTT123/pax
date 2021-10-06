from pax._src.module import Module, PaxFieldKind
from pax._src.rng import next_rng_key, seed_rng_key
from pax._src.side_effects import jit_, pmap_, vmap_
from pax._src.transforms import (
    apply_gradients,
    apply_gradients_,
    apply_mp_policy,
    apply_updates,
    enable_eval_mode,
    enable_train_mode,
    flatten_module,
    forward,
    freeze_parameters,
    scan_bugs,
    select_kind,
    select_parameters,
    select_states,
    transform_gradients,
    transform_gradients_,
    unfreeze_parameters,
    update_parameters,
    update_states,
)
from pax._src.utils import dropout, grad_parameters, scan

from . import initializers, nets, nn, utils

__all__ = [
    "apply_gradients_",
    "apply_gradients",
    "apply_mp_policy",
    "apply_updates",
    "dropout",
    "enable_eval_mode",
    "enable_train_mode",
    "flatten_module",
    "forward",
    "freeze_parameters",
    "grad_parameters",
    "grad",
    "initializers",
    "jit_",
    "Module",
    "nets",
    "next_rng_key",
    "nn",
    "PaxFieldKind",
    "pmap_",
    "scan_bugs",
    "scan",
    "seed_rng_key",
    "select_kind",
    "select_parameters",
    "select_states",
    "transform_gradients_",
    "transform_gradients",
    "unfreeze_parameters",
    "update_parameters",
    "update_states",
    "utils",
    "vmap_",
]

try:
    del _src  # pylint: disable=undefined-variable
except NameError:
    pass
