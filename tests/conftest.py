"""Pytest configuration."""

import os

try:
    # pylint: disable=unused-import
    import hypothesis  # noqa F401
    from hypothesis import settings
except ImportError:
    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True

# ignore test files associated with hypothesis strategies
collect_ignore = []
if not HAS_HYPOTHESIS:
    collect_ignore.append("test_strategies.py")
else:
    settings.register_profile("ci", max_examples=100, deadline=20000)
    settings.register_profile("dev", max_examples=10, deadline=2000)
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
