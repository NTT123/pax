"""PAX package."""

from pax import experimental, nets, utils
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
from pax._src.nn import (
    EMA,
    GRU,
    LSTM,
    BatchNorm1D,
    BatchNorm2D,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Embed,
    GroupNorm,
    GRUState,
    Identity,
    Lambda,
    LayerNorm,
    Linear,
    LSTMState,
    MultiHeadAttention,
    RngSeq,
    Sequential,
    avg_pool,
    max_pool,
)
from pax._src.nn.dropout import dropout
from pax._src.utils import build_update_fn, grad, scan, value_and_grad

__version__ = "0.5.1.dev"

__all__ = (
    "apply_mp_policy",
    "assert_structure_equal",
    "avg_pool",
    "BatchNorm1D",
    "BatchNorm2D",
    "build_update_fn",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "dropout",
    "Dropout",
    "EMA",
    "Embed",
    "enable_eval_mode",
    "enable_train_mode",
    "experimental",
    "freeze_parameters",
    "grad",
    "GroupNorm",
    "GRU",
    "GRUState",
    "Identity",
    "Lambda",
    "LayerNorm",
    "Linear",
    "LSTM",
    "LSTMState",
    "max_pool",
    "module_and_value",
    "Module",
    "MultiHeadAttention",
    "nets",
    "next_rng_key",
    "nn",
    "ParameterModule",
    "parameters_method",
    "pure",
    "RngSeq",
    "scan",
    "seed_rng_key",
    "select_parameters",
    "Sequential",
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
