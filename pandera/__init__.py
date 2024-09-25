# pylint: disable=wrong-import-position
"""A flexible and expressive pandas validation library."""

import os
import platform

from pandera._patch_numpy2 import _patch_numpy2

_patch_numpy2()

import pandera.backends
import pandera.backends.base.builtin_checks
import pandera.backends.base.builtin_hypotheses
import pandera.backends.pandas
from pandera import errors, external_config
from pandera.api import extensions
from pandera.api.checks import Check
from pandera.api.dataframe.model_components import (
    Field,
    check,
    dataframe_check,
    dataframe_parser,
    parser,
)
from pandera.api.hypotheses import Hypothesis
from pandera.api.pandas.array import SeriesSchema
from pandera.api.pandas.components import Column, Index, MultiIndex
from pandera.api.pandas.container import DataFrameSchema
from pandera.api.pandas.model import DataFrameModel
from pandera.api.parsers import Parser
from pandera.decorators import check_input, check_io, check_output, check_types
from pandera.dtypes import (
    Bool,
    Category,
    Complex,
    Complex64,
    Complex128,
    DataType,
    Date,
    DateTime,
    Decimal,
    Float,
    Float16,
    Float32,
    Float64,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    String,
    Timedelta,
    Timestamp,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from pandera.engines.numpy_engine import Object
from pandera.engines.pandas_engine import (
    BOOL,
    INT8,
    INT16,
    INT32,
    INT64,
    PANDAS_1_2_0_PLUS,
    PANDAS_1_3_0_PLUS,
    PANDAS_2_0_0_PLUS,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    pandas_version,
)
from pandera.schema_inference.pandas import infer_schema
from pandera.version import __version__

if platform.system() != "Windows":
    # pylint: disable=ungrouped-imports
    from pandera.dtypes import Complex256, Float128


__all__ = [
    # dtypes
    "Bool",
    "Category",
    "Complex",
    "Complex64",
    "Complex128",
    "Complex256",
    "Date",
    "DataType",
    "DateTime",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Float128",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "String",
    "Timedelta",
    "Timestamp",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    # numpy_engine
    "Object",
    # pandas_engine
    "BOOL",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "PANDAS_1_2_0_PLUS",
    "PANDAS_1_3_0_PLUS",
    "PANDAS_2_0_0_PLUS",
    "STRING",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    # pandera.engines.pandas_engine
    "pandas_version",
    # checks
    "Check",
    # parsers
    "Parser",
    # decorators
    "check_input",
    "check_io",
    "check_output",
    "check_types",
    # hypotheses
    "Hypothesis",
    # model
    "DataFrameModel",
    # model_components
    "Field",
    "check",
    "dataframe_check",
    "parser",
    "dataframe_parser",
    # schema_components
    "Column",
    "Index",
    "MultiIndex",
    # schema_inference
    "infer_schema",
    # schemas
    "DataFrameSchema",
    "SeriesSchema",
    # version
    "__version__",
]
