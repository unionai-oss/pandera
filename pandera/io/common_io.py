"""Shared helpers for emitting schema/model Python scripts across backends."""

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import Literal

from pandera.engines import pandas_engine
from pandera.io._minimal import COLUMN_DEFAULTS, DF_SCHEMA_DEFAULTS

_FORMAT_SCRIPT_WARNING_MESSAGE = (
    "Schema script formatting requires 'black' to be installed. "
    "Please install 'black' to use this feature."
)

BackendId = Literal["pandas", "polars", "ibis", "pyspark"]

SCRIPT_TEMPLATE = """
from pandera import (
    DataFrameSchema, Column, Check, Index, MultiIndex
)

schema = DataFrameSchema(
    columns={{{columns}}},
    checks={checks},
    index={index},
    dtype={dtype},
    coerce={coerce},
    strict={strict},
    name={name},
    ordered={ordered},
    unique={unique},
    report_duplicates={report_duplicates},
    unique_column_names={unique_column_names},
    add_missing_columns={add_missing_columns},
    title={title},
    description={description},
)
"""

COLUMN_TEMPLATE = """
{qual}Column(
    dtype={dtype},
    checks={checks},
    nullable={nullable},
    unique={unique},
    coerce={coerce},
    required={required},
    regex={regex},
    description={description},
    title={title},
)
"""

INDEX_TEMPLATE = """
Index(
    dtype={dtype},
    checks={checks},
    nullable={nullable},
    coerce={coerce},
    name={name},
    description={description},
    title={title},
)
"""

MULTIINDEX_TEMPLATE = """
MultiIndex(indexes=[{indexes}])
"""


def _get_dtype_string_alias(dtype: pandas_engine.DataType) -> str:
    """String alias of the datatype for script emission (pandas engine)."""
    str_alias = str(dtype)
    try:
        pandas_engine.Engine.dtype(str_alias)
    except TypeError as e:  # pragma: no cover
        raise TypeError(
            f"string alias {str_alias} for datatype "
            f"'{dtype.__module__}.{dtype.__class__.__name__}' not "
            "recognized."
        ) from e
    return f'"{dtype}"'


def _schema_script_qual(backend: str) -> str:
    return "" if backend == "pandas" else "pa."


def _get_dataframe_schema_statistics_fn(backend: str):
    if backend == "pandas":
        from pandera.schema_statistics.pandas import (
            get_dataframe_schema_statistics as gfs,
        )
    elif backend == "polars":
        from pandera.schema_statistics.polars import (
            get_dataframe_schema_statistics as gfs,
        )
    elif backend == "ibis":
        from pandera.schema_statistics.ibis import (
            get_dataframe_schema_statistics as gfs,
        )
    elif backend == "pyspark":
        from pandera.schema_statistics.pyspark import (
            get_dataframe_schema_statistics as gfs,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return gfs


def _schema_script_imports(backend: str) -> str:
    if backend == "pandas":
        return (
            "from pandera import (\n"
            "    DataFrameSchema, Column, Check, Index, MultiIndex\n"
            ")\n\n"
        )
    if backend == "polars":
        return "import pandera.polars as pa\n\n"
    if backend == "ibis":
        return "import pandera.ibis as pa\n\n"
    if backend == "pyspark":
        return "import pandera.pyspark as pa\n\n"
    raise ValueError(f"unknown backend {backend!r}")


def _format_checks(checks_list):
    """Format checks into string representation including options."""
    if checks_list is None:
        return "None"

    checks = []
    for check_kwargs in checks_list:
        if check_kwargs is None:
            warnings.warn(
                "Check cannot be serialized. This check will be ignored"
            )
            continue

        options = (
            check_kwargs.pop("options", {})
            if isinstance(check_kwargs, dict)
            else {}
        )

        if "check_name" not in options:
            warnings.warn(
                "Check cannot be serialized. This check will be ignored"
            )
            continue

        check_name = options.pop("check_name")

        if isinstance(check_kwargs, dict):
            args = ", ".join(
                f"{k}={v.__repr__()}" for k, v in check_kwargs.items()
            )
        else:
            args = check_kwargs.__repr__()

        if options:
            if args:
                args += ", "
            args += ", ".join(
                f"{k}={v.__repr__()}" for k, v in options.items()
            )

        checks.append(f"Check.{check_name}({args})")

    return f"[{', '.join(checks)}]"


def _format_index(index_statistics):
    index = []
    for properties in index_statistics:
        dtype = properties.get("dtype")
        description = properties.get("description")
        title = properties.get("title")
        index_code = INDEX_TEMPLATE.format(
            dtype=(None if dtype is None else _get_dtype_string_alias(dtype)),
            checks=(
                "None"
                if properties["checks"] is None
                else _format_checks(properties["checks"])
            ),
            nullable=properties["nullable"],
            coerce=properties["coerce"],
            name=(
                "None"
                if properties["name"] is None
                else f'"{properties["name"]}"'
            ),
            description=(None if description is None else f'"{description}"'),
            title=(None if title is None else f'"{title}"'),
        )
        index.append(index_code.strip())

    if len(index) == 1:
        return index[0]

    return MULTIINDEX_TEMPLATE.format(indexes=",".join(index)).strip()


def _format_column_minimal(column, *, qual: str = ""):
    """Build ``Column(...)`` source with only non-default kwargs."""
    from pandera.schema_statistics.pandas import parse_checks

    parts = []
    if column.dtype is not None:
        parts.append(f"dtype={_get_dtype_string_alias(column.dtype)}")
    pc = parse_checks(column.checks)
    if pc:
        parts.append(f"checks={_format_checks(pc)}")
    for attr in COLUMN_DEFAULTS:
        if attr == "name":
            continue
        if not hasattr(column, attr):
            continue
        val = getattr(column, attr)
        if val == COLUMN_DEFAULTS[attr]:
            continue
        parts.append(f"{attr}={val!r}")
    if column.metadata:
        parts.append(f"metadata={column.metadata!r}")
    if not parts:
        return f"{qual}Column()"
    inner = ",\n    ".join(parts)
    return f"{qual}Column(\n    {inner}\n)"


def _to_script_minimal(dataframe_schema, *, backend: str = "pandas"):
    """Build ``DataFrameSchema(...)`` source with only non-default kwargs."""
    from pandera.schema_statistics.pandas import parse_checks

    qual = _schema_script_qual(backend)
    get_stats = _get_dataframe_schema_statistics_fn(backend)
    columns = {
        name: _format_column_minimal(column, qual=qual)
        for name, column in dataframe_schema.columns.items()
    }
    column_str = ", ".join(f"{k!r}: {v}" for k, v in columns.items())
    parts = [f"columns={{{column_str}}}"]
    pc = parse_checks(dataframe_schema.checks)
    if pc:
        parts.append(f"checks={_format_checks(pc)}")
    stats = get_stats(dataframe_schema)
    if stats["index"] is not None:
        parts.append(f"index={_format_index(stats['index'])}")
    for key in DF_SCHEMA_DEFAULTS:
        val = getattr(dataframe_schema, key)
        if val == DF_SCHEMA_DEFAULTS[key]:
            continue
        if key == "dtype" and val is not None:
            parts.append(f"dtype={_get_dtype_string_alias(val)}")
        else:
            parts.append(f"{key}={val!r}")
    if getattr(dataframe_schema, "metadata", None):
        parts.append(f"metadata={dataframe_schema.metadata!r}")
    if getattr(dataframe_schema, "drop_invalid_rows", False):
        parts.append(
            f"drop_invalid_rows={dataframe_schema.drop_invalid_rows!r}"
        )
    inner = ",\n    ".join(parts)
    body = f"schema = {qual}DataFrameSchema(\n    {inner}\n)"
    return f"{_schema_script_imports(backend)}{body}"


def _format_script(script: str) -> str:
    try:
        import black
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_FORMAT_SCRIPT_WARNING_MESSAGE) from exc

    formatter = partial(black.format_str, mode=black.FileMode(line_length=80))
    return formatter(script)


def to_script(
    dataframe_schema,
    path_or_buf=None,
    *,
    minimal: bool = True,
    script_type: Literal["schema", "model"] = "schema",
    backend: BackendId = "pandas",
):
    """Emit a schema or model script (internal: pass ``backend`` for each API).

    IO submodules expose backend-specific :func:`to_script` wrappers without a
    ``backend`` argument.
    """
    from pandera.io._script_model import to_dataframe_model_script

    if script_type == "model":
        script = to_dataframe_model_script(
            dataframe_schema, backend, minimal=minimal
        ).strip()
        if "Timedelta" in script:
            script = "from pandas import Timedelta\n" + script
        if "Timestamp" in script:
            script = "from pandas import Timestamp\n" + script
        formatted_script = _format_script(script)
        if path_or_buf is None:
            return formatted_script
        with Path(path_or_buf).open("w", encoding="utf-8") as f:
            f.write(formatted_script)
        return None

    if minimal:
        script = _to_script_minimal(dataframe_schema, backend=backend).strip()
        if "Timedelta" in script:
            script = "from pandas import Timedelta\n" + script
        if "Timestamp" in script:
            script = "from pandas import Timestamp\n" + script
        formatted_script = _format_script(script)
        if path_or_buf is None:
            return formatted_script
        with Path(path_or_buf).open("w", encoding="utf-8") as f:
            f.write(formatted_script)
        return None

    get_stats = _get_dataframe_schema_statistics_fn(backend)
    statistics = get_stats(dataframe_schema)
    qual = _schema_script_qual(backend)

    columns = {}
    for colname, properties in statistics["columns"].items():
        dtype = properties.get("dtype")
        description = properties["description"]
        title = properties["title"]
        column_code = COLUMN_TEMPLATE.format(
            qual=qual,
            dtype=(None if dtype is None else _get_dtype_string_alias(dtype)),
            checks=_format_checks(properties["checks"]),
            nullable=properties["nullable"],
            unique=properties["unique"],
            coerce=properties["coerce"],
            required=properties["required"],
            regex=properties["regex"],
            description=(None if description is None else f'"{description}"'),
            title=(None if title is None else f'"{title}"'),
        )
        columns[colname] = column_code.strip()

    index = (
        None
        if statistics["index"] is None
        else _format_index(statistics["index"])
    )

    column_str = ", ".join(f"'{k}': {v}" for k, v in columns.items())

    script = SCRIPT_TEMPLATE.format(
        columns=column_str,
        checks=statistics["checks"],
        index=index,
        dtype=dataframe_schema.dtype,
        coerce=dataframe_schema.coerce,
        strict=dataframe_schema.strict,
        name=dataframe_schema.name.__repr__(),
        ordered=dataframe_schema.ordered,
        unique=dataframe_schema.unique,
        report_duplicates=f'"{dataframe_schema.report_duplicates}"',
        unique_column_names=dataframe_schema.unique_column_names,
        add_missing_columns=dataframe_schema.add_missing_columns,
        title=dataframe_schema.title,
        description=dataframe_schema.description,
    ).strip()

    if backend != "pandas":
        script = script.replace(
            "from pandera import (\n"
            "    DataFrameSchema, Column, Check, Index, MultiIndex\n"
            ")\n\n",
            _schema_script_imports(backend),
        )
        script = script.replace(
            "schema = DataFrameSchema(",
            f"schema = {qual}DataFrameSchema(",
        )

    if "Timedelta" in script:
        script = "from pandas import Timedelta\n" + script
    if "Timestamp" in script:
        script = "from pandas import Timestamp\n" + script

    formatted_script = _format_script(script)

    if path_or_buf is None:
        return formatted_script

    with Path(path_or_buf).open("w", encoding="utf-8") as f:
        f.write(formatted_script)
    return None
