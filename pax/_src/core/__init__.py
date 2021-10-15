"""PAX Module"""

from .flatten_module import flatten_module
from .mixed_precision import apply_mp_policy
from .module import Module, PaxKind
from .module_and_value import module_and_value
from .pure import pure
from .transforms import (
    enable_eval_mode,
    enable_train_mode,
    freeze_parameters,
    select_parameters,
    select_states,
    unfreeze_parameters,
    update_parameters,
    update_states,
)
from .utils import assert_structure_equal
