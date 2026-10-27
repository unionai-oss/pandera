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
        # strategy tests build whole dataframes (e.g. size-5 frames with
        # regex-named columns) via ``st.data()``; the smallest natural input
        # can legitimately be large, which intermittently trips this size-based
        # health check. See the suppression guidance in the ``FailedHealthCheck``
        # message for ``large_base_example``. This is unrelated to the code under
        # test, so it's suppressed rather than worked around in individual tests.
        hypothesis.HealthCheck.large_base_example,
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
