"""PAX Module"""

from .graph_module import GraphModule, InputNode, build_graph_module
from .mixed_precision import apply_mp_policy, unwrap_mp_policy
from .module import EmptyNode, Module, parameters_method
from .module_and_value import module_and_value
from .mutable import mutable
from .pure import pure, purecall
from .transforms import (
    enable_eval_mode,
    enable_train_mode,
    freeze_parameters,
    select_parameters,
    unfreeze_parameters,
    update_parameters,
)
from .utility_modules import Flattener, LazyModule, ParameterModule, StateModule
from .utils import assert_structure_equal
