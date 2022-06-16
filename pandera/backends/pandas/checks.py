from pandera.backends.base import BaseCheckBackend

# TODO: This module should:
# 1. implement the pandas-specific check object-processing logic
# 2. use the user-facing registration API for all of the built-in checks


class PandasCheckBackend(BaseCheckBackend):
    pass


class PandasCheckFieldBackend(PandasCheckBackend):
    pass


class PandasCheckContainerBackend(PandasCheckBackend):
    pass
