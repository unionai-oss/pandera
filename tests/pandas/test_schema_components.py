"""Testing the components of the Schema objects."""

import copy
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pytest

from pandera.pandas import (
    Check,
    Column,
    DataFrameSchema,
    DateTime,
    Float,
    Index,
    Int,
    MultiIndex,
    SeriesSchema,
    String,
    errors,
)
from pandera.engines.pandas_engine import Engine, pandas_version


def test_column() -> None:
    """Test that the Column object can be used to check dataframe."""
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [2.0, 3.0, 4.0],
            "c": ["foo", "bar", "baz"],
        }
    )

    column_a = Column(Int, name="a")
    column_b = Column(Float, name="b")
    column_c = Column(String, name="c")

    assert isinstance(
        data.pipe(column_a).pipe(column_b).pipe(column_c), pd.DataFrame
    )

    with pytest.raises(errors.SchemaError):
        Column(Int)(data)


def test_column_coerce() -> None:
    """Test that the Column object can be used to coerce dataframe types."""
    data = pd.DataFrame({"a": [1, 2, 3]})
    column_schema = Column(Int, name="a", coerce=True)
    validated = column_schema.validate(data)
    assert isinstance(validated, pd.DataFrame)
    assert Engine.dtype(validated.a.dtype) == Engine.dtype(int)


def test_column_in_dataframe_schema():
    """Test that a Column check returns a dataframe."""
    schema = DataFrameSchema(
        {"a": Column(Int, Check(lambda x: x > 0, element_wise=True))}
    )
    data = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(schema.validate(data), pd.DataFrame)


def test_index_schema():
    """Tests that when specifying a DataFrameSchema Index pandera validates
    and errors appropriately."""
    schema = DataFrameSchema(
        index=Index(
            Int,
            [
                Check(lambda x: 1 <= x <= 11, element_wise=True),
                Check(lambda index: index.mean() > 1),
            ],
        )
    )
    df = pd.DataFrame(index=range(1, 11), dtype="int64")
    assert isinstance(schema.validate(df), pd.DataFrame)
    assert schema.index.names == [None]

    with pytest.raises(errors.SchemaError):
        schema.validate(pd.DataFrame(index=range(1, 20)))


@pytest.mark.parametrize("dtype", [Float, Int, String])
def test_index_schema_coerce(dtype):
    """Test that index can be type-coerced."""
    schema = DataFrameSchema(index=Index(dtype, coerce=True))
    df = pd.DataFrame(index=pd.Index([1, 2, 3, 4], dtype="int64"))
    validated_index_dtype = Engine.dtype(schema(df).index.dtype)
    assert schema.index.dtype.check(validated_index_dtype)


@pytest.mark.parametrize("dtype", [Float, Int, String])
def test_index_schema_coerce_when_coerce_specified_at_schema_level(dtype):
    """Test that index can be type-coerced when coercion requested at schema level"""
    schema = DataFrameSchema(index=Index(dtype), coerce=True)
    df = pd.DataFrame(index=pd.Index([1, 2, 3, 4], dtype="int64"))
    validated_index_dtype = Engine.dtype(schema(df).index.dtype)
    assert schema.index.dtype.check(validated_index_dtype)


def test_multi_index_columns() -> None:
    """Tests that multi-index Columns within DataFrames validate correctly."""
    schema = DataFrameSchema(
        {
            ("zero", "foo"): Column(Float, Check(lambda s: (s > 0) & (s < 1))),
            ("zero", "bar"): Column(
                String, Check(lambda s: s.isin(["a", "b", "c", "d"]))
            ),
            ("one", "foo"): Column(Int, Check(lambda s: (s > 0) & (s < 10))),
            ("one", "bar"): Column(
                DateTime, Check(lambda s: s == pd.Timestamp(2019, 1, 1))
            ),
        }
    )
    validated_df = schema.validate(
        pd.DataFrame(
            {
                ("zero", "foo"): [0.1, 0.2, 0.7, 0.3],
                ("zero", "bar"): ["a", "b", "c", "d"],
                ("one", "foo"): [1, 6, 4, 7],
                ("one", "bar"): pd.to_datetime(["2019/01/01"] * 4),
            }
        )
    )
    assert isinstance(validated_df, pd.DataFrame)


@pytest.mark.parametrize(
    "schema,df",
    [
        [
            DataFrameSchema({("foo", "baz"): Column(int)}),
            pd.DataFrame({("foo", "baz"): ["a", "b", "c"]}),
        ],
        [
            DataFrameSchema(
                {
                    ("foo", "bar"): Column(
                        int, checks=Check(lambda s: s == 1)
                    ),
                }
            ),
            pd.DataFrame({("foo", "bar"): [1, 2, 3, 4, 5]}),
        ],
    ],
)
def test_multi_index_column_errors(schema, df) -> None:
    """
    Test that schemas with MultiIndex columns correctly raise SchemaErrors on
    lazy validation.
    """
    with pytest.raises(errors.SchemaErrors):
        schema.validate(df, lazy=True)


def test_multi_index_index() -> None:
    """Tests that multi-index Indexes within DataFrames validate correctly."""
    schema = DataFrameSchema(
        columns={
            "column1": Column(Float, Check(lambda s: s > 0)),
            "column2": Column(Float, Check(lambda s: s > 0)),
        },
        index=MultiIndex(
            indexes=[
                Index(Int, Check(lambda s: (s < 5) & (s >= 0)), name="index0"),
                Index(
                    String,
                    Check(lambda s: s.isin(["foo", "bar"])),
                    name="index1",
                ),
            ]
        ),
    )

    df = pd.DataFrame(
        data={
            "column1": [0.1, 0.5, 123.1, 10.6, 22.31],
            "column2": [0.1, 0.5, 123.1, 10.6, 22.31],
        },
        index=pd.MultiIndex.from_arrays(
            [[0, 1, 2, 3, 4], ["foo", "bar", "foo", "bar", "foo"]],
            names=["index0", "index1"],
        ),
    )

    validated_df = schema.validate(df)
    assert isinstance(validated_df, pd.DataFrame)
    assert schema.index.names == ["index0", "index1"]

    # failure case
    df_fail = df.copy()
    df_fail.index = pd.MultiIndex.from_arrays(
        [[-1, 1, 2, 3, 4], ["foo", "bar", "foo", "bar", "foo"]],
        names=["index0", "index1"],
    )
    with pytest.raises(errors.SchemaError):
        schema.validate(df_fail)


def test_single_index_multi_index_mismatch() -> None:
    """Tests the failure case that attempting to validate a MultiIndex DataFrame
    against a single index schema raises a SchemaError with a constructive error
    message."""
    ind = pd.MultiIndex.from_tuples(
        [("a", "b"), ("c", "d"), ("e", "f")],
        names=("one", "two"),
    )
    df_fail = pd.DataFrame(index=ind)
    schema = DataFrameSchema(index=Index(name="key"))

    with pytest.raises(errors.SchemaError):
        schema.validate(df_fail)


def test_multi_index_schema_coerce() -> None:
    """Test that multi index can be type-coerced."""
    indexes = [
        Index(Float),
        Index(Int),
        Index(String),
    ]
    schema = DataFrameSchema(index=MultiIndex(indexes=indexes))
    df = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [
                [1.0, 2.1, 3.5, 4.8],
                [5, 6, 7, 8],
                ["9", "10", "11", "12"],
            ]
        )
    )
    validated_df = schema(df)
    for level_i in range(validated_df.index.nlevels):
        index_dtype = validated_df.index.get_level_values(level_i).dtype
        assert indexes[level_i].dtype.check(Engine.dtype(index_dtype))


def tests_multi_index_subindex_coerce() -> None:
    """MultIndex component should override sub indexes."""
    indexes = [
        Index(String, coerce=True),
        Index(String, coerce=False),
        Index(String, coerce=True),
        Index(String, coerce=False),
    ]

    data = pd.DataFrame(index=pd.MultiIndex.from_arrays([[1, 2, 3, 4]] * 4))

    # coerce=True in MultiIndex and DataFrameSchema should override subindex
    # coerce setting
    for schema_override in [
        DataFrameSchema(index=MultiIndex(indexes, coerce=True)),
        DataFrameSchema(index=MultiIndex(indexes), coerce=True),
    ]:
        validated_df_override = schema_override(data)
        for level_i in range(validated_df_override.index.nlevels):
            assert (
                validated_df_override.index.get_level_values(level_i).dtype
                == "object"
            )

    # coerce=False at the MultiIndex level should result in two type errors
    schema = DataFrameSchema(index=MultiIndex(indexes))
    with pytest.raises(errors.SchemaErrors) as e:
        schema(data, lazy=True)

    assert len(e.value.message["SCHEMA"]) == 1


@pytest.mark.skipif(
    pandas_version().release <= (1, 3, 5),
    reason="MultiIndex dtypes are buggy prior to pandas 1.4.*",
)
@pytest.mark.parametrize("coerce", [True, False])
def tests_multi_index_subindex_coerce_with_empty_subindex(coerce) -> None:
    """MultIndex component should override each sub indexes dtype,
    even if the sub indexes are empty (ie do not rely on
    numpy to infer the subindex dtype.
    """
    indexes = [
        Index(pd.Int64Dtype, coerce=coerce),
        Index(pd.StringDtype, coerce=coerce),
    ]

    data = pd.DataFrame(index=pd.MultiIndex.from_arrays([[]] * len(indexes)))
    schema_override = DataFrameSchema(index=MultiIndex(indexes))

    if coerce:
        validated_df_override = schema_override(data)
        for level_i in range(validated_df_override.index.nlevels):
            assert isinstance(
                validated_df_override.index.get_level_values(level_i).dtype,
                type(indexes[level_i].dtype.type),  # type: ignore[attr-defined]
            )
    else:
        with pytest.raises(
            errors.SchemaErrors,
        ) as e:
            schema_override(data, lazy=True)

        assert len(e.value.message["SCHEMA"]) == 1


def test_schema_component_equality_operators():
    """Test the usage of == for Column, Index and MultiIndex."""
    column = Column(Int, Check(lambda s: s >= 0))
    index = Index(Int, [Check(lambda x: 1 <= x <= 11, element_wise=True)])
    multi_index = MultiIndex(
        indexes=[
            Index(Int, Check(lambda s: (s < 5) & (s >= 0)), name="index0"),
            Index(
                String, Check(lambda s: s.isin(["foo", "bar"])), name="index1"
            ),
        ]
    )
    not_equal_schema = DataFrameSchema(
        {"col1": Column(Int, Check(lambda s: s >= 0))}
    )

    assert column == copy.deepcopy(column)
    assert column != not_equal_schema
    assert index == copy.deepcopy(index)
    assert index != not_equal_schema
    assert multi_index == copy.deepcopy(multi_index)
    assert multi_index != not_equal_schema


def test_column_regex() -> None:
    """Test that column regex work on single-level column index."""
    column_schema = Column(
        Int, Check(lambda s: s >= 0), name="foo_*", regex=True
    )

    dataframe_schema = DataFrameSchema(
        {
            "foo_*": Column(Int, Check(lambda s: s >= 0), regex=True),
        }
    )

    data = pd.DataFrame(
        {
            "foo_1": range(10),
            "foo_2": range(10, 20),
            "foo_3": range(20, 30),
            "bar_1": range(10),
            "bar_2": range(10, 20),
            "bar_3": range(20, 30),
        }
    )
    assert isinstance(column_schema.validate(data), pd.DataFrame)
    assert isinstance(dataframe_schema.validate(data), pd.DataFrame)

    # Raise an error on multi-index column case
    data.columns = pd.MultiIndex.from_tuples(
        (
            ("foo_1", "biz_1"),
            ("foo_2", "baz_1"),
            ("foo_3", "baz_2"),
            ("bar_1", "biz_2"),
            ("bar_2", "biz_3"),
            ("bar_3", "biz_3"),
        )
    )
    with pytest.raises(IndexError):
        column_schema.validate(data)
    with pytest.raises(IndexError):
        dataframe_schema.validate(data)


def test_column_regex_multiindex() -> None:
    """Text that column regex works on multi-index column."""
    column_schema = Column(
        Int,
        Check(lambda s: s >= 0),
        name=("foo_*", "baz_*"),
        regex=True,
    )
    dataframe_schema = DataFrameSchema(
        {
            ("foo_*", "baz_*"): Column(
                Int, Check(lambda s: s >= 0), regex=True
            ),
        }
    )

    data = pd.DataFrame(
        {
            ("foo_1", "biz_1"): range(10),
            ("foo_2", "baz_1"): range(10, 20),
            ("foo_3", "baz_2"): range(20, 30),
            ("bar_1", "biz_2"): range(10),
            ("bar_2", "biz_3"): range(10, 20),
            ("bar_3", "biz_3"): range(20, 30),
        }
    )
    assert isinstance(column_schema.validate(data), pd.DataFrame)
    assert isinstance(dataframe_schema.validate(data), pd.DataFrame)

    # Raise an error if tuple column name is applied to a dataframe with a
    # flat pd.Index object.
    failure_column_cases = (
        [f"foo_{i}" for i in range(6)],
        pd.MultiIndex.from_tuples(
            [(f"foo_{i}", f"bar_{i}", f"baz_{i}") for i in range(6)]
        ),
    )
    for columns in failure_column_cases:
        data.columns = columns  # type: ignore
        with pytest.raises(IndexError):
            column_schema.validate(data)
        with pytest.raises(IndexError):
            dataframe_schema.validate(data)


@pytest.mark.parametrize(
    "column_name_regex, expected_matches, error",
    (
        # match all values in first level, only baz_* for second level
        ((".", "baz_.+"), [("foo_2", "baz_1"), ("foo_3", "baz_2")], None),
        # match bar_* in first level, all values in second level
        (
            ("bar_.+", "."),
            [("bar_1", "biz_2"), ("bar_2", "biz_3"), ("bar_3", "biz_3")],
            None,
        ),
        # match specific columns in both levels
        (("foo_.+", "baz_.+"), [("foo_2", "baz_1"), ("foo_3", "baz_2")], None),
        (("foo_.+", "^biz_1$"), [("foo_1", "biz_1")], None),
        (("^foo_3$", "^baz_2$"), [("foo_3", "baz_2")], None),
        # no matches should raise a SchemaError
        (("fiz", "."), None, errors.SchemaError),
        # using a string name for a multi-index column raises IndexError
        ("foo_1", None, IndexError),
        # mis-matching number of elements in a tuple column name raises
        # IndexError
        (("foo_.+",), None, IndexError),
        (("foo_.+", ".", "."), None, IndexError),
        (("foo_.+", ".", ".", "."), None, IndexError),
    ),
)
def test_column_regex_matching(
    column_name_regex: str,
    expected_matches: Optional[List[Tuple[str, str]]],
    error: Type[BaseException],
) -> None:
    """
    Column regex pattern matching should yield correct matches and raise
    expected errors.
    """
    columns = pd.MultiIndex.from_tuples(
        (
            ("foo_1", "biz_1"),
            ("foo_2", "baz_1"),
            ("foo_3", "baz_2"),
            ("bar_1", "biz_2"),
            ("bar_2", "biz_3"),
            ("bar_3", "biz_3"),
        )
    )
    check_obj = pd.DataFrame(columns=columns)

    column_schema = Column(
        Int,
        Check(lambda s: s >= 0),
        name=column_name_regex,
        regex=True,
    )
    if error is not None:
        with pytest.raises(error):
            column_schema.get_regex_columns(check_obj)
    else:
        matched_columns = column_schema.get_regex_columns(check_obj)
        assert expected_matches == matched_columns.tolist()


def test_column_regex_error_failure_cases():

    data = pd.DataFrame({"a": [0, 2], "b": [1, 3]})

    column_schema = Column(
        name=r"a|b",
        dtype=int,
        regex=True,
        checks=Check(
            element_wise=True,
            name="custom_check",
            check_fn=lambda *args, **kwargs: False,
        ),
    )

    expected_error = pd.DataFrame(
        {
            "schema_context": ["Column"] * 4,
            "column": ["a", "a", "b", "b"],
            "check": ["custom_check"] * 4,
            "check_number": [0] * 4,
            "failure_case": [0, 2, 1, 3],
            "index": [0, 1, 0, 1],
        }
    )

    try:
        column_schema.validate(data, lazy=True)
    except errors.SchemaErrors as err:
        pd.testing.assert_frame_equal(err.failure_cases, expected_error)


INT_REGEX = r"-?\d+$"
FLOAT_REGEX = r"-?\d+\.\d+$"
DATETIME_REGEX = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


@pytest.mark.parametrize(
    "column_name_regex, expected_matches",
    [
        # match all
        [".+", [1, 2.2, 3.1415, -1, -3.6, pd.Timestamp("2018/01/01")]],
        # match integers
        [INT_REGEX, [1, -1]],
        # match floats
        [FLOAT_REGEX, [2.2, 3.1415, -3.6]],
        # match datetimes
        [DATETIME_REGEX, [pd.Timestamp("2018/01/01")]],
    ],
)
def test_column_regex_matching_non_str_types(
    column_name_regex: str, expected_matches: List
) -> None:
    """Non-string column names should be cast into str for regex matching."""
    columns = pd.Index([1, 2.2, 3.1415, -1, -3.6, pd.Timestamp("2018/01/01")])
    check_obj = pd.DataFrame(columns=columns)
    column_schema = Column(name=column_name_regex, regex=True)
    matched_columns = column_schema.get_regex_columns(check_obj)
    assert expected_matches == [*matched_columns]


@pytest.mark.parametrize(
    "column_name_regex, expected_matches",
    [
        # match all
        [
            (".+", ".+"),
            [
                ("foo", 1),
                ("foo", pd.Timestamp("2018/01/01")),
                (1, 2.2),
                (3.14, -1),
            ],
        ],
        # match (str, int)
        [("foo", INT_REGEX), [("foo", 1)]],
        # match (str, pd.Timestamp)
        [("foo", DATETIME_REGEX), [("foo", pd.Timestamp("2018/01/01"))]],
        # match (int, float)
        [(INT_REGEX, FLOAT_REGEX), [(1, 2.2)]],
        # match (float, int)
        [(FLOAT_REGEX, INT_REGEX), [(3.14, -1)]],
    ],
)
def test_column_regex_matching_non_str_types_multiindex(
    column_name_regex: Tuple[str, str], expected_matches: List[Tuple[Any, Any]]
) -> None:
    """
    Non-string column names should be cast into str for regex matching in
    MultiIndex column case.
    """
    columns = pd.MultiIndex.from_tuples(
        (
            ("foo", 1),
            ("foo", pd.Timestamp("2018/01/01")),
            (1, 2.2),
            (3.14, -1),
        )
    )
    check_obj = pd.DataFrame(columns=columns)
    column_schema = Column(name=column_name_regex, regex=True)
    matched_columns = column_schema.get_regex_columns(check_obj)
    assert expected_matches == [*matched_columns]


def test_column_regex_strict() -> None:
    """Test that Column regex patterns correctly parsed in DataFrameSchema."""
    data = pd.DataFrame(
        {
            "foo_1": [1, 2, 3],
            "foo_2": [1, 2, 3],
            "foo_3": [1, 2, 3],
        }
    )
    schema = DataFrameSchema(
        columns={"foo_*": Column(Int, regex=True)}, strict=True
    )
    assert isinstance(schema.validate(data), pd.DataFrame)

    # adding an extra column in the dataframe should cause error
    data = data.assign(bar=[1, 2, 3])
    with pytest.raises(errors.SchemaError):
        schema.validate(data)

    # adding an extra regex column to the schema should pass the strictness
    # test
    validated_data = schema.add_columns(
        {"bar_*": Column(Int, regex=True)}
    ).validate(data.assign(bar_1=[1, 2, 3]))
    assert isinstance(validated_data, pd.DataFrame)


def test_column_regex_non_str_types() -> None:
    """Check that column name regex matching excludes non-string types."""
    data = pd.DataFrame(
        {
            1: [1, 2, 3],
            2.2: [1, 2, 3],
            pd.Timestamp("2018/01/01"): [1, 2, 3],
            "foo_1": [1, 2, 3],
            "foo_2": [1, 2, 3],
            "foo_3": [1, 2, 3],
        }
    )
    schema = DataFrameSchema(
        columns={
            "foo_": Column(Int, Check.gt(0), regex=True),
            r"\d+": Column(Int, Check.gt(0), regex=True),
            r"\d+\.\d+": Column(Int, Check.gt(0), regex=True),
            "2018-01-01": Column(Int, Check.gt(0), regex=True),
        },
    )
    assert isinstance(schema.validate(data), pd.DataFrame)

    # test MultiIndex column case
    data = pd.DataFrame(
        {
            (1, 1): [1, 2, 3],
            (2.2, 4.5): [1, 2, 3],
            ("foo", "bar"): [1, 2, 3],
        }
    )
    schema = DataFrameSchema(
        columns={("foo_*", "bar_*"): Column(Int, regex=True)},
    )
    schema.validate(data)


def test_column_type_can_be_set() -> None:
    """Test that the Column dtype can be edited during schema construction."""

    column_a = Column(Int, name="a")
    changed_type = Float

    column_a.dtype = Float  # type: ignore [assignment]

    assert column_a.dtype == Engine.dtype(changed_type)

    for invalid_dtype in ("foobar", "bar"):
        with pytest.raises(TypeError):
            column_a.dtype = invalid_dtype  # type: ignore [assignment]

    for invalid_dtype in (1, 2.2, ["foo", 1, 1.1], {"b": 1}):
        with pytest.raises(TypeError):
            column_a.dtype = invalid_dtype  # type: ignore [assignment]


@pytest.mark.parametrize(
    "multiindex, error",
    [
        [
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], [1, 2, 3]], names=["a", "a"]
            ),
            False,
        ],
        [
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], ["a", "b", "c"]], names=["a", "a"]
            ),
            True,
        ],
    ],
)
@pytest.mark.parametrize(
    "schema",
    [
        MultiIndex([Index(int, name="a"), Index(int, name="a")]),
        MultiIndex([Index(int, name="a"), Index(int, name="a")], coerce=True),
    ],
)
def test_multiindex_duplicate_index_names(
    multiindex: pd.MultiIndex, error: bool, schema: MultiIndex
) -> None:
    """Test MultiIndex schema component can handle duplicate index names."""
    if error:
        with pytest.raises(errors.SchemaError):
            schema(pd.DataFrame(index=multiindex))
        with pytest.raises(errors.SchemaErrors):
            schema(pd.DataFrame(index=multiindex), lazy=True)
    else:
        assert isinstance(schema(pd.DataFrame(index=multiindex)), pd.DataFrame)


@pytest.mark.parametrize(
    "multiindex, schema, error",
    [
        # No index names
        [
            pd.MultiIndex.from_arrays([[1], [1]]),
            MultiIndex([Index(int), Index(int)]),
            False,
        ],
        # index names on pa.MultiIndex, no index name on schema
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=["a", "b"]),
            MultiIndex([Index(int), Index(int)]),
            True,
        ],
        # no index names on pa.MultiIndex, index names on schema
        [
            pd.MultiIndex.from_arrays([[1], [1]]),
            MultiIndex([Index(int, name="a"), Index(int, name="b")]),
            True,
        ],
        # mixed index names and None
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=["a", None]),
            MultiIndex([Index(int, name="a"), Index(int)]),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=[None, "b"]),
            MultiIndex([Index(int), Index(int, name="b")]),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=["b", None]),
            MultiIndex([Index(int, name="a"), Index(int)]),
            True,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=[None, "a"]),
            MultiIndex([Index(int), Index(int, name="b")]),
            True,
        ],
        # duplicated index names
        [
            pd.MultiIndex.from_arrays([[1], [1], [1]], names=["a", "a", None]),
            MultiIndex([Index(int, name="a"), Index(int)]),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1], [1]], names=["a", "a", None]),
            MultiIndex([Index(int, name="a"), Index(int)], coerce=True),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1], [1]], names=["a", None, "a"]),
            MultiIndex([Index(int, name="a"), Index(int)]),
            "column 'a' out-of-order",
        ],
        [
            pd.MultiIndex.from_arrays(
                [[1], [1], [1], [1]], names=["a", "a", None, None]
            ),
            MultiIndex([Index(int, name="a"), Index(int)]),
            False,
        ],
        [
            pd.MultiIndex.from_arrays(
                [[1], [1], [1], [1]], names=["a", "a", None, None]
            ),
            MultiIndex([Index(int, name="a"), Index(int)], coerce=True),
            False,
        ],
    ],
)
def test_multiindex_ordered(
    multiindex: pd.MultiIndex, schema: MultiIndex, error: bool
) -> None:
    """Test that MultiIndex schema checks index name order."""
    if error:
        with pytest.raises(
            errors.SchemaError, match=error if isinstance(error, str) else None
        ):
            schema(pd.DataFrame(index=multiindex))
        with pytest.raises(errors.SchemaErrors):
            schema(pd.DataFrame(index=multiindex), lazy=True)
        return
    assert isinstance(schema(pd.DataFrame(index=multiindex)), pd.DataFrame)


@pytest.mark.parametrize(
    "multiindex, schema, error",
    [
        # unordered schema component, no names in multiindex
        [
            pd.MultiIndex.from_arrays([[1], [1]]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")], ordered=False
            ),
            True,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=[None, "b"]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")], ordered=False
            ),
            True,
        ],
        # unordered schema component with names in multiindex
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=["b", "a"]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")], ordered=False
            ),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1]], names=["b", "a"]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")],
                ordered=False,
                coerce=True,
            ),
            False,
        ],
        # unordered schema component with duplicated names in multiindex and
        # dtype coercion
        [
            pd.MultiIndex.from_arrays([[1], [1], [1]], names=["b", "a", "a"]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")],
                ordered=False,
            ),
            False,
        ],
        [
            pd.MultiIndex.from_arrays([[1], [1], [1]], names=["b", "a", "a"]),
            MultiIndex(
                [Index(int, name="a"), Index(int, name="b")],
                coerce=True,
                ordered=False,
            ),
            False,
        ],
    ],
)
def test_multiindex_unordered(
    multiindex: pd.MultiIndex, schema: MultiIndex, error: bool
) -> None:
    """Test MultiIndex schema unordered validation."""
    if error:
        with pytest.raises(errors.SchemaError):
            schema(pd.DataFrame(index=multiindex))
        with pytest.raises(errors.SchemaErrors):
            schema(pd.DataFrame(index=multiindex), lazy=True)
        return
    assert isinstance(schema(pd.DataFrame(index=multiindex)), pd.DataFrame)


@pytest.mark.parametrize(
    "indexes",
    [
        [Index(int)],
        [Index(int, name="a"), Index(int)],
        [Index(int), Index(int, name="a")],
    ],
)
def test_multiindex_unordered_init_exception(indexes: List[Index]) -> None:
    """Un-named indexes in unordered MultiIndex raises an exception."""
    with pytest.raises(errors.SchemaInitError):
        MultiIndex(indexes, ordered=False)


@pytest.mark.parametrize(
    "indexes",
    [
        [Column(int)],
        [Column(int, name="a"), Index(int)],
        [Index(int), Column(int, name="a")],
        [SeriesSchema(int)],
        1,
        1.0,
        "foo",
    ],
)
def test_multiindex_incorrect_input(indexes) -> None:
    """Passing in non-Index object raises SchemaInitError."""
    with pytest.raises((errors.SchemaInitError, TypeError)):
        MultiIndex(indexes)


def test_index_validation_pandas_string_dtype():
    """Test that pandas string type is correctly validated."""

    if pandas_version().release <= (1, 3, 5):
        pytest.xfail(
            "pd.StringDtype is not supported in the pd.Index in pandas<=1.3.5"
        )

    schema = DataFrameSchema(
        columns={"data": Column(int)},
        index=Index(pd.StringDtype(), name="uid"),
    )

    df = pd.DataFrame(
        {"data": range(2)},
        index=pd.Index(["one", "two"], dtype=pd.StringDtype(), name="uid"),
    )

    assert isinstance(schema.validate(df), pd.DataFrame)


@pytest.fixture()
def xfail_int_with_nans(request):
    dtype = request.getfixturevalue("dtype")
    input_value = request.getfixturevalue("input_value")
    coerce = request.getfixturevalue("coerce")

    if dtype == "Int64" and input_value is not None and not coerce:
        pytest.xfail("NaN is considered a Float64")


@pytest.mark.parametrize(
    "dtype,default",
    [
        (str, "a default"),
        (bool, True),
        (bool, False),
        (float, 42.0),
        ("Int64", 0),
    ],
)
@pytest.mark.parametrize("input_value", [None, np.nan])
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.usefixtures("xfail_int_with_nans")
def test_column_default_works_when_dtype_match(
    input_value: Any, coerce: bool, dtype: Any, default: Any
):
    """Test ``default`` fills ``nan`` values as expected when the ``dtype`` matches that of the ``Column``"""
    column = Column(dtype, name="column1", default=default, coerce=coerce)
    df = pd.DataFrame({"column1": [input_value]})
    column.validate(df, inplace=True)

    assert df.iloc[0]["column1"] == default


@pytest.mark.parametrize(
    "dtype,default",
    [
        (str, 1),
        (bool, 42.0),
        (float, True),
        ("Int64", "a default"),
    ],
)
def test_column_default_errors_on_dtype_mismatch(dtype: Any, default: Any):
    """Test that setting a ``default`` of different ``dtype`` to that of the ```Column`` raises an error"""
    column = Column(dtype, name="column1", default=default)
    df = pd.DataFrame({"column1": [None]})

    with pytest.raises(errors.SchemaError):
        column.validate(df, inplace=True)
