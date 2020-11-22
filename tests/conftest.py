"""Pytest configuration."""

try:
    # pylint: disable=unused-import
    import hypothesis
except ImportError:
    HAS_HYPOTHESIS = False
else:
    HAS_HYPOTHESIS = True

# ignore test files associated with hypothesis strategies
collect_ignore = []
if not HAS_HYPOTHESIS:
    collect_ignore.append("test_strategies.py")
