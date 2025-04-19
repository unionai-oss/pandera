"""Register pandas backends."""

from functools import lru_cache
from typing import Optional

from pandera.backends.pandas.array import SeriesSchemaBackend
from pandera.backends.pandas.checks import PandasCheckBackend
from pandera.backends.pandas.components import (
    ColumnBackend,
    IndexBackend,
    MultiIndexBackend,
)
from pandera.backends.pandas.container import DataFrameSchemaBackend
from pandera.backends.pandas.hypotheses import PandasHypothesisBackend
from pandera.backends.pandas.parsers import PandasParserBackend


@lru_cache
def register_pandas_backends(
    check_cls_fqn: Optional[str] = None,
):  # pylint: disable=unused-argument
    """Register pandas backends.

    This function is called at schema initialization in the _register_*_backends
    method.

    :param framework_name: name of the framework to register backends for.
        Allowable types are "pandas", "dask", "modin", "pyspark", and
        "geopandas".
    """

    # pylint: disable=import-outside-toplevel,unused-import,cyclic-import
    from pandera._patch_numpy2 import _patch_numpy2

    _patch_numpy2()

    from pandera.api.checks import Check
    from pandera.api.hypotheses import Hypothesis
    from pandera.api.pandas.array import SeriesSchema
    from pandera.api.pandas.components import Column, Index, MultiIndex
    from pandera.api.pandas.container import DataFrameSchema
    from pandera.api.parsers import Parser
    from pandera.api.pandas.types import get_backend_types

    # NOTE: This registers the deprecated DataFrameSchema class. Remove this
    # once the deprecated class is removed.
    from pandera._pandas_deprecated import (
        DataFrameSchema as _DataFrameSchemaDeprecated,
    )

    assert check_cls_fqn is not None, (
        "pandas backend registration requires passing in the fully qualified "
        "check class name"
    )
    backend_types = get_backend_types(check_cls_fqn)

    from pandera.backends.pandas import builtin_checks, builtin_hypotheses

    for t in backend_types.check_backend_types:
        Check.register_backend(t, PandasCheckBackend)
        Hypothesis.register_backend(t, PandasHypothesisBackend)
        Parser.register_backend(t, PandasParserBackend)

    for t in backend_types.dataframe_datatypes:
        DataFrameSchema.register_backend(t, DataFrameSchemaBackend)
        _DataFrameSchemaDeprecated.register_backend(t, DataFrameSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.series_datatypes:
        SeriesSchema.register_backend(t, SeriesSchemaBackend)
        Column.register_backend(t, ColumnBackend)
        MultiIndex.register_backend(t, MultiIndexBackend)
        Index.register_backend(t, IndexBackend)

    for t in backend_types.index_datatypes:
        Index.register_backend(t, IndexBackend)

    for t in backend_types.multiindex_datatypes:
        MultiIndex.register_backend(t, MultiIndexBackend)
