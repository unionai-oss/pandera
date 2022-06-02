from pandera.core import base

# TODO:
# - core.checks implement the check specification and registry
# - backend.checks implement:
#   - check pre/post-processing, e.g. query, groupby, aggregate
#   - built-in checks
#   - check result handling (maybe even error formatting?)


class BaseCheck(base.BaseCheck):
    ...


class Check(BaseCheck):
    ...


class Hypothesis(BaseCheck):
    ...
