"""PAX package."""

from pax import experimental, nets, nn, utils
from pax._src.core import (
    Module,
    ParameterModule,
    StateModule,
    apply_mp_policy,
    assert_structure_equal,
    enable_eval_mode,
    enable_train_mode,
    freeze_parameters,
    module_and_value,
    parameters_method,
    pure,
    select_parameters,
    unfreeze_parameters,
    unwrap_mp_policy,
    update_parameters,
)
from pax._src.core.rng import next_rng_key, seed_rng_key
from pax._src.nn.dropout import dropout
from pax._src.utils import build_update_fn, grad, scan, value_and_grad

__version__ = "0.5.0.dev"

__all__ = (
    "apply_mp_policy",
    "assert_structure_equal",
    "build_update_fn",
    "dropout",
    "enable_eval_mode",
    "enable_train_mode",
    "experimental",
    "freeze_parameters",
    "grad",
    "module_and_value",
    "Module",
    "nets",
    "next_rng_key",
    "nn",
    "ParameterModule",
    "parameters_method",
    "pure",
    "scan",
    "seed_rng_key",
    "select_parameters",
    "StateModule",
    "unfreeze_parameters",
    "unwrap_mp_policy",
    "update_parameters",
    "utils",
    "value_and_grad",
)

try:
    del _src  # pylint: disable=undefined-variable
except NameError:
    pass
