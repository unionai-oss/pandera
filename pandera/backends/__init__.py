"""Pandera backends."""

# ensure that base builtin checks and hypothesis are registered
import pandera.backends.base.builtin_checks
import pandera.backends.base.builtin_hypotheses
import pandera.backends.pandas
