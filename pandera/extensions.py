"""Extensions module, for backwards compatibility."""

# pylint: disable=unused-import
from pandera.core.extensions import (
    register_builtin_check,
    register_builtin_hypothesis,
    generate_check_signature,
    generate_check_annotations,
    modify_check_fn_doc,
    update_check_fn_proxy,
    CheckType,
    register_check_method,
    register_check_statistics,
)
