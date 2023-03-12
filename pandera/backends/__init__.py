"""Pandera backends."""

# ensure that base builtin checks and hypothesis are registered
import pandera.backends.stubs.builtin_checks
import pandera.backends.stubs.builtin_hypotheses

import pandera.backends.pandas
