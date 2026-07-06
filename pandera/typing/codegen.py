"""Generate static typing helpers for DataFrameModel column access.

Pyright/Pylance cannot infer column types from ``DataFrame[Schema]`` alone
because column names are defined dynamically on the model class. This module
generates a typed :class:`~pandera.typing.pandas.DataFrame` subclass with
``__getitem__`` overloads and column attributes that static analyzers can
understand.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pandera.api.dataframe.model import DataFrameModel as BaseDataFrameModel
from pandera.typing.common import IndexBase, SeriesBase

_SERIES_NAMES = {
    "Series",
    "GeoSeries",
}


def _is_model_field(name: str) -> bool:
    return not name.startswith("_") and name != "Config"


def _series_dtype(annotation: Any) -> str:
    """Return the source representation of a column's ``Series[T]`` dtype."""
    origin = get_origin(annotation)
    if origin is Annotated:
        annotation = get_args(annotation)[0]
        origin = get_origin(annotation)

    args = get_args(annotation)
    if args:
        dtype = args[0]
        if isinstance(dtype, type):
            return dtype.__name__
        return str(dtype).replace("typing.", "")

    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _column_series_type(
    annotation: Any, *, series_import: str = "Series"
) -> str:
    """Return a source string for the Series type of a model field annotation."""
    origin = get_origin(annotation)
    raw = annotation
    if origin is Annotated:
        raw = get_args(annotation)[0]
        origin = get_origin(raw)

    if (
        origin is not None
        and getattr(origin, "__name__", None) in _SERIES_NAMES
    ):
        return f"{series_import}[{_series_dtype(raw)}]"

    if inspect.isclass(raw) and issubclass(raw, (SeriesBase, IndexBase)):
        return f"{series_import}[{_series_dtype(raw)}]"

    if isinstance(raw, type):
        return f"{series_import}[{raw.__name__}]"
    return series_import


def generate_typed_dataframe_source(
    model: type[BaseDataFrameModel],
    *,
    class_name: str | None = None,
    dataframe_module: str = "pandera.typing",
) -> str:
    """Generate source code for a Pyright-compatible typed DataFrame subclass.

    Parameters
    ----------
    model:
        A :class:`~pandera.api.dataframe.model.DataFrameModel` subclass.
    class_name:
        Name for the generated class. Defaults to ``{ModelName}DataFrame``.
    dataframe_module:
        Import path used for ``DataFrame`` and ``Series`` in the generated
        source.

    Returns
    -------
    str
        Python source code defining a typed dataframe class.

    Examples
    --------
    >>> from pandera.typing import Series
    >>> import pandera.pandas as pa
    >>> class Schema(pa.DataFrameModel):
    ...     a: Series[int]
    >>> print(generate_typed_dataframe_source(Schema))  # doctest: +SKIP
    """
    if not isinstance(model, type) or not issubclass(
        model, BaseDataFrameModel
    ):
        raise TypeError(
            f"Expected a DataFrameModel subclass, found {type(model)!r}."
        )

    typed_name = class_name or f"{model.__name__}DataFrame"
    hints = get_type_hints(model, include_extras=True)
    columns = {
        name: ann for name, ann in hints.items() if _is_model_field(name)
    }

    if not columns:
        raise ValueError(
            f"Model {model.__name__!r} has no column fields to generate types for."
        )

    overload_lines = []
    attr_lines = []
    for col_name, annotation in columns.items():
        series_type = _column_series_type(annotation)
        attr_lines.append(f"    {col_name}: {series_type}")
        overload_lines.extend(
            [
                "    @overload",
                f"    def __getitem__(self, key: Literal[{col_name!r}]) -> {series_type}: ...",
            ]
        )

    overload_lines.extend(
        [
            "    @overload",
            "    def __getitem__(self, key: str) -> Series: ...",
            "    def __getitem__(self, key: str) -> Series:",
            "        return cast(Series, super().__getitem__(key))",
        ]
    )

    lines = [
        "from typing import Literal, cast, overload",
        "",
        f"import {dataframe_module.rsplit('.', 1)[0]}.pandas as pa",
        f"from {dataframe_module} import DataFrame, Series",
        "",
        "",
        f"class {typed_name}(DataFrame[{model.__name__}]):",
        '    """Typed dataframe wrapper for static analysis (Pyright/Pylance)."""',
        "",
        *attr_lines,
        "",
        *overload_lines,
        "",
    ]
    return textwrap.dedent("\n".join(lines))
