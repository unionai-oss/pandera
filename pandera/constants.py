"""Pandera constants."""

CHECK_OUTPUT_KEY = "check_output"
CHECK_OUTPUT_SUFFIX = f"__{CHECK_OUTPUT_KEY}__"
FAILURE_CASE_KEY = "failure_case"

#: Actions supported by the ``on_missing`` column option and the
#: ``on_missing_columns`` schema option. Only ``"warn"`` is available for now,
#: but this is kept as a collection so more actions can be added later.
ON_MISSING_ACTIONS = ("warn",)
