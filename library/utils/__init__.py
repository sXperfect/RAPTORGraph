from .core import (
    exists,
    default,
    identity,
    cycle,
    ensure_dict,
    has_int_squareroot,
    num_to_groups,
)

from .data import (
    conv_vec_to_cat,
    _get_apply_dispatcher,
    lists_to_boolean_matrix,
    bool_matrix_to_embed_ids
)

from .datasets import (
    remove_ctrl
)

from .tensor import (
    l2norm,
    extract
)