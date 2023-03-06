"""Pandera backends."""

# ensure that base builtin checks and hypothesis are registered
import pandera.core.base.builtin_checks
import pandera.core.base.builtin_hypotheses

import pandera.backends.pandas
