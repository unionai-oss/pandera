from pandera.backends.base import BaseCheckBackend


class PandasCheckBackend(BaseCheckBackend):
    pass


class PandasCheckFieldBackend(PandasCheckBackend):
    pass


class PandasCheckContainerBackend(PandasCheckBackend):
    pass
