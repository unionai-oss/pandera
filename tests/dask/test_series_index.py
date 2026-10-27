"""Index validation for Dask Series."""

import dask.dataframe as dd
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.parsers import Parser


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "index_schema,index",
    [
        (pa.Index(int, unique=True), [1, 2, 2]),
        (pa.Index(int), ["a", "b", "c"]),
        (pa.Index(int, checks=pa.Check.ge(0)), [-1, 0, 1]),
    ],
    ids=["unique", "dtype", "check"],
)
def test_series_index_validation(index_schema, index, lazy, inplace):
    """Index failures are deferred until the Dask graph is computed."""
    schema = pa.SeriesSchema(float, index=index_schema)
    series = pd.Series(
        [1.0, 2.0, 3.0], index=index, name=f"values_{lazy}_{inplace}"
    )
    dseries = dd.from_pandas(series, npartitions=1)
    error = pa.errors.SchemaErrors if lazy else pa.errors.SchemaError

    with pytest.raises(error):
        schema.validate(series, lazy=lazy, inplace=inplace)

    validated = schema.validate(dseries, lazy=lazy, inplace=inplace)
    assert isinstance(validated, dd.Series)
    with pytest.raises(error):
        validated.compute()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_series_valid_index(lazy, inplace):
    """Valid index constraints preserve the Series and its schema."""
    schema = pa.SeriesSchema(
        float,
        index=pa.Index(int, unique=True, checks=pa.Check.ge(0)),
    )
    series = pd.Series(
        [1.0, 2.0, 3.0], index=[1, 2, 3], name=f"values_{lazy}_{inplace}"
    )
    dseries = dd.from_pandas(series, npartitions=2)

    validated = schema.validate(dseries, lazy=lazy, inplace=inplace)

    assert validated.pandera.schema == schema
    pd.testing.assert_series_equal(validated.compute(), series)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "schema,series,expected",
    [
        (
            pa.SeriesSchema(int, coerce=True, index=pa.Index(int)),
            pd.Series(["1", "2", "3"]),
            pd.Series([1, 2, 3]),
        ),
        (
            pa.SeriesSchema(
                int,
                parsers=Parser(lambda series: series + 1),
                index=pa.Index(int),
            ),
            pd.Series([1, 2, 3]),
            pd.Series([2, 3, 4]),
        ),
    ],
    ids=["coerce", "parser"],
)
def test_series_index_preserves_transformed_values(
    schema, series, expected, inplace
):
    """Index validation must retain the values from Series validation."""
    dseries = dd.from_pandas(series, npartitions=2)

    validated = schema.validate(dseries, inplace=inplace)

    pd.testing.assert_series_equal(validated.compute(), expected)
    pd.testing.assert_series_equal(
        schema.validate(series, inplace=inplace), expected
    )


def test_series_index_validation_in_later_partition():
    """Every partition validates its index when the graph is computed."""
    schema = pa.SeriesSchema(int, index=pa.Index(int, pa.Check.ge(0)))
    dseries = dd.from_pandas(
        pd.Series([1, 2, 3, 4], index=[0, 1, 2, -1]),
        npartitions=2,
        sort=False,
    )

    validated = schema.validate(dseries)

    with pytest.raises(pa.errors.SchemaError):
        validated.compute()


def test_series_index_coercion():
    """Computing a Series applies coercion to both its values and index."""
    schema = pa.SeriesSchema(
        int, coerce=True, index=pa.Index(int, coerce=True)
    )
    series = pd.Series(["1", "2", "3"], index=["4", "5", "6"])
    dseries = dd.from_pandas(series, npartitions=2)

    validated = schema.validate(dseries)

    pd.testing.assert_series_equal(
        validated.compute(), pd.Series([1, 2, 3], index=[4, 5, 6])
    )
