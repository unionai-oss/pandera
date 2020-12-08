"""Testing the components of the Schema objects."""

import copy

import numpy as np
import pandas as pd
import pytest

from pandera import (
    Check,
    Column,
    DataFrameSchema,
    DateTime,
    Float,
    Index,
    Int,
    MultiIndex,
    Object,
    String,
    errors,
)
from tests.test_dtypes import TESTABLE_DTYPES


def test_column():
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


def test_coerce_nullable_object_column():
    """Test that Object dtype coercing preserves object types."""
    df_objects_with_na = pd.DataFrame(
        {"col": [1, 2.0, [1, 2, 3], {"a": 1}, np.nan, None]}
    )

    column_schema = Column(Object, name="col", coerce=True, nullable=True)

    validated_df = column_schema.validate(df_objects_with_na)
    assert isinstance(validated_df, pd.DataFrame)
    assert pd.isna(validated_df["col"].iloc[-1])
    assert pd.isna(validated_df["col"].iloc[-2])
    for i in range(4):
        isinstance(
            validated_df["col"].iloc[i],
            type(df_objects_with_na["col"].iloc[i]),
        )


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

    with pytest.raises(errors.SchemaError):
        schema.validate(pd.DataFrame(index=range(1, 20)))


@pytest.mark.parametrize("pdtype", [Float, Int, String, String])
def test_index_schema_coerce(pdtype):
    """Test that index can be type-coerced."""
    schema = DataFrameSchema(index=Index(pdtype, coerce=True))
    df = pd.DataFrame(index=pd.Index([1, 2, 3, 4], dtype="int64"))
    validated_df = schema(df)
    # pandas-native "string" dtype doesn't apply to indexes
    assert (
        validated_df.index.dtype == "object"
        if pdtype is String
        else pdtype.str_alias
    )


def test_multi_index_columns():
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


def test_multi_index_index():
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

    # failure case
    df_fail = df.copy()
    df_fail.index = pd.MultiIndex.from_arrays(
        [[-1, 1, 2, 3, 4], ["foo", "bar", "foo", "bar", "foo"]],
        names=["index0", "index1"],
    )
    with pytest.raises(errors.SchemaError):
        schema.validate(df_fail)


def test_multi_index_schema_coerce():
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
        assert (
            validated_df.index.get_level_values(level_i).dtype
            == indexes[level_i].dtype
        )


def tests_multi_index_subindex_coerce():
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
    with pytest.raises(
        errors.SchemaErrors, match="A total of 2 schema errors were found"
    ):
        schema(data, lazy=True)


@pytest.mark.parametrize("pandas_dtype, expected", TESTABLE_DTYPES)
def test_column_dtype_property(pandas_dtype, expected):
    """Tests that the dtypes provided by Column match pandas dtypes"""
    assert Column(pandas_dtype).dtype == expected


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


def test_column_regex():
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


def test_column_regex_multiindex():
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
        data.columns = columns
        with pytest.raises(IndexError):
            column_schema.validate(data)
        with pytest.raises(IndexError):
            dataframe_schema.validate(data)


@pytest.mark.parametrize(
    "column_name_regex, expected_matches, error",
    (
        # match all values in first level, only baz_* for second level
        ((".", "baz_*"), [("foo_2", "baz_1"), ("foo_3", "baz_2")], None),
        # match bar_* in first level, all values in second level
        (
            ("bar_*", "."),
            [("bar_1", "biz_2"), ("bar_2", "biz_3"), ("bar_3", "biz_3")],
            None,
        ),
        # match specific columns in both levels
        (("foo_*", "baz_*"), [("foo_2", "baz_1"), ("foo_3", "baz_2")], None),
        (("foo_*", "^biz_1$"), [("foo_1", "biz_1")], None),
        (("^foo_3$", "^baz_2$"), [("foo_3", "baz_2")], None),
        # no matches should raise a SchemaError
        (("fiz", "."), None, errors.SchemaError),
        # using a string name for a multi-index column raises IndexError
        ("foo_1", None, IndexError),
        # mis-matching number of elements in a tuple column name raises IndexError
        (("foo_*",), None, IndexError),
        (("foo_*", ".", "."), None, IndexError),
        (("foo_*", ".", ".", "."), None, IndexError),
    ),
)
def test_column_regex_matching(column_name_regex, expected_matches, error):
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

    column_schema = Column(
        Int,
        Check(lambda s: s >= 0),
        name=column_name_regex,
        regex=True,
    )
    if error is not None:
        with pytest.raises(error):
            column_schema.get_regex_columns(columns)
    else:
        matched_columns = column_schema.get_regex_columns(columns)
        assert expected_matches == matched_columns.tolist()


def test_column_regex_strict():
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


def test_column_regex_non_str_types():
    """Check that column name regex matching excludes non-string types."""
    data = pd.DataFrame(
        {
            1: [1, 2, 3],
            2.2: [1, 2, 3],
            pd.Timestamp("2018/01/01"): [1, 2, 3],
            "foo": [1, 2, 3],
        }
    )
    schema = DataFrameSchema(
        columns={"foo_*": Column(Int, regex=True)},
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


@pytest.mark.parametrize("column_key", [1, 100, 0.543])
def test_non_str_column_name_regex(column_key):
    """Check that Columns with non-str names cannot have regex=True."""

    with pytest.raises(ValueError):
        DataFrameSchema(
            {
                column_key: Column(
                    Float,
                    checks=Check.greater_than_or_equal_to(0),
                    regex=True,
                ),
            }
        )

    with pytest.raises(ValueError):
        Column(
            Float,
            checks=Check.greater_than_or_equal_to(0),
            name=column_key,
            regex=True,
        )


def test_column_type_can_be_set():
    """Test that the Column dtype can be edited during schema construction."""

    column_a = Column(Int, name="a")
    changed_type = Float

    column_a.pandas_dtype = Float

    assert column_a.pandas_dtype == changed_type
    assert column_a.dtype == changed_type.str_alias

    for invalid_dtype in ("foobar", "bar"):
        with pytest.raises(TypeError):
            column_a.pandas_dtype = invalid_dtype

    for invalid_dtype in (1, 2.2, ["foo", 1, 1.1], {"b": 1}):
        with pytest.raises(TypeError):
            column_a.pandas_dtype = invalid_dtype


@pytest.mark.parametrize(
    "multiindex, error",
    [
        [
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], [1, 2, 3]], names=["a", "a"]
            ),
            None,
        ],
        [
            pd.MultiIndex.from_arrays(
                [[1, 2, 3], ["a", "b", "c"]], names=["a", "a"]
            ),
            errors.SchemaError,
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
def test_multiindex_duplicate_index_names(multiindex, error, schema):
    """Test MultiIndex schema component can handle duplicate index names."""
    if error is None:
        assert isinstance(schema(pd.DataFrame(index=multiindex)), pd.DataFrame)
    else:
        with pytest.raises(error):
            schema(pd.DataFrame(index=multiindex))
