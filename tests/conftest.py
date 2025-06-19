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
    suppressed_health_checks = [
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.too_slow,
        hypothesis.HealthCheck.filter_too_much,
    ]

    settings.register_profile(
        "ci",
        max_examples=10,
        deadline=None,
        suppress_health_check=suppressed_health_checks,
    )
    settings.register_profile(
        "dev",
        max_examples=30,
        deadline=None,
        suppress_health_check=suppressed_health_checks,
    )
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
