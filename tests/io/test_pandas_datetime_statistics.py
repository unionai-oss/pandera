"""Datetime check statistics retain their validation semantics on persistence."""

import datetime
import json

import pandas as pd
import pytest

import pandera.extensions as extensions
import pandera.pandas as pa
from pandera.engines import pandas_engine
from pandera.errors import SchemaErrorReason


@pytest.mark.parametrize("format", ["json", "yaml"])
@pytest.mark.parametrize("minimal", [False, True])
@pytest.mark.parametrize("component", ["column", "index"])
@pytest.mark.parametrize("check_name", ["ge", "eq", "in_range"])
@pytest.mark.parametrize(
    "threshold",
    [
        pd.Timestamp("2026-01-01 00:00:00.500000001"),
        datetime.datetime(2026, 1, 1, microsecond=500001),
        pd.Timestamp("2026-01-01 00:00:00.500000001", tz="Asia/Shanghai"),
        datetime.datetime(
            2026,
            1,
            1,
            microsecond=500001,
            tzinfo=datetime.timezone(datetime.timedelta(hours=5, minutes=30)),
        ),
    ],
)
def test_datetime_check_statistics_roundtrip(
    format, minimal, component, check_name, threshold
):
    """Subsecond and timezone-aware bounds keep accepting the same values."""
    if format == "yaml":
        pytest.importorskip("yaml")
    dtype = pandas_engine.DateTime(tz=threshold.tzinfo)
    if check_name == "in_range":
        check = pa.Check.in_range(
            threshold, threshold + datetime.timedelta(milliseconds=100)
        )
    else:
        check = getattr(pa.Check, check_name)(threshold)
    if component == "column":
        schema = pa.DataFrameSchema({"ts": pa.Column(dtype, check)})
    else:
        schema = pa.DataFrameSchema(index=pa.Index(dtype, check, name="ts"))

    serialized = getattr(schema, f"to_{format}")(minimal=minimal)
    loaded = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)

    loaded_check = (
        loaded.columns["ts"].checks[0]
        if component == "column"
        else loaded.index.checks[0]
    )
    for offset in [-1, 0, 1, 50000000, 100000000, 100000001]:
        value = pd.Timestamp(threshold) + pd.Timedelta(offset, unit="ns")
        if component == "column":
            frame = pd.DataFrame({"ts": pd.Series([value], dtype=dtype.type)})
        else:
            frame = pd.DataFrame(
                index=pd.DatetimeIndex([value], dtype=dtype.type, name="ts")
            )
        if check_name == "ge":
            valid = offset >= 0
        elif check_name == "eq":
            valid = offset == 0
        else:
            valid = 0 <= offset <= 100000000
        for validator in [schema, loaded]:
            if valid:
                validator.validate(frame)
            else:
                with pytest.raises(pa.errors.SchemaError) as exc_info:
                    validator.validate(frame)
                assert (
                    exc_info.value.reason_code
                    == SchemaErrorReason.DATAFRAME_CHECK
                )
    assert loaded_check.statistics == check.statistics


@pytest.mark.parametrize("format", ["json", "yaml"])
@pytest.mark.parametrize("minimal", [False, True])
@pytest.mark.parametrize("hour", [5, 6])
def test_datetime_bounds_preserve_instants_across_dst(format, minimal, hour):
    """ISO bounds preserve the instant and offset, not the named zone."""
    if format == "yaml":
        pytest.importorskip("yaml")
    threshold = pd.Timestamp(
        f"2026-11-01 {hour:02}:30:00.500000001", tz="UTC"
    ).tz_convert("America/New_York")
    dtype = pandas_engine.DateTime(tz="America/New_York")
    schema = pa.DataFrameSchema(
        {"ts": pa.Column(dtype, pa.Check.ge(threshold))}
    )

    serialized = getattr(schema, f"to_{format}")(minimal=minimal)
    loaded = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)
    loaded_bound = loaded.columns["ts"].checks[0].statistics["min_value"]

    assert loaded_bound.value == threshold.value
    assert loaded_bound.utcoffset() == threshold.utcoffset()
    for offset in [-1, 0, 1]:
        value = threshold + pd.Timedelta(offset, unit="ns")
        frame = pd.DataFrame({"ts": pd.Series([value], dtype=dtype.type)})
        for validator in [schema, loaded]:
            if offset < 0:
                with pytest.raises(pa.errors.SchemaError):
                    validator.validate(frame)
            else:
                validator.validate(frame)


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_legacy_whole_second_datetime_check_statistics(format):
    """Existing whole-second schema snapshots remain readable."""
    snapshot = {
        "schema_type": "dataframe",
        "columns": {
            "ts": {
                "dtype": "datetime64[ns]",
                "checks": {"greater_than_or_equal_to": "2026-01-01 00:00:00"},
            }
        },
    }
    if format == "yaml":
        yaml = pytest.importorskip("yaml")
        serialized = yaml.safe_dump(snapshot)
    else:
        serialized = json.dumps(snapshot)

    schema = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)

    assert schema.columns["ts"].checks[0].statistics == {
        "min_value": pd.Timestamp("2026-01-01")
    }
    schema.validate(
        pd.DataFrame({"ts": pd.Series(["2026-01-01"], dtype="datetime64[ns]")})
    )
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(
            pd.DataFrame(
                {"ts": pd.Series(["2025-12-31"], dtype="datetime64[ns]")}
            )
        )


@pytest.mark.parametrize("format", ["json", "yaml"])
@pytest.mark.parametrize("minimal", [False, True])
@pytest.mark.parametrize(
    "label",
    [
        "2026",
        "2026-01-01",
        "2026-01-01 00:00:00.500000001",
        "2026-01-01 00:00:00.500000001+05:30",
    ],
)
@pytest.mark.parametrize("timezone", [None, "UTC"])
def test_datetime_check_non_timestamp_string_statistics(
    format, minimal, label, timezone, monkeypatch
):
    """Custom date-looking string statistics keep their original type."""
    if format == "yaml":
        pytest.importorskip("yaml")
    monkeypatch.setattr(
        pa.Check,
        "REGISTERED_CUSTOM_CHECKS",
        dict(pa.Check.REGISTERED_CUSTOM_CHECKS),
    )

    @extensions.register_check_method(statistics=["label"])
    def datetime_label_check(series, *, label):
        return label == expected_label

    expected_label = label
    dtype = pandas_engine.DateTime(tz=timezone)
    schema = pa.DataFrameSchema(
        {"ts": pa.Column(dtype, pa.Check.datetime_label_check(label))}
    )

    serialized = getattr(schema, f"to_{format}")(minimal=minimal)
    loaded = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)

    assert loaded.columns["ts"].checks[0].statistics == {"label": label}
    frame = pd.DataFrame(
        {
            "ts": pd.Series(
                [pd.Timestamp("2026-01-01", tz=timezone)], dtype=dtype.type
            )
        }
    )
    schema.validate(frame)
    loaded.validate(frame)


@pytest.mark.parametrize("format", ["json", "yaml"])
@pytest.mark.parametrize(
    "threshold",
    [
        pd.Timestamp("2026-01-01 00:00:00.500000001"),
        datetime.datetime(2026, 1, 1, microsecond=500001),
    ],
)
def test_custom_datetime_statistics_keep_legacy_encoding(
    format, threshold, monkeypatch
):
    """The comparison-bound fix does not change custom check stat decoding."""
    if format == "yaml":
        pytest.importorskip("yaml")
    monkeypatch.setattr(
        pa.Check,
        "REGISTERED_CUSTOM_CHECKS",
        dict(pa.Check.REGISTERED_CUSTOM_CHECKS),
    )

    @extensions.register_check_method(statistics=["label"])
    def datetime_typed_label_check(series, *, label):
        return isinstance(label, datetime.datetime) and label.year == 2026

    schema = pa.DataFrameSchema(
        {
            "ts": pa.Column(
                "datetime64[ns]",
                pa.Check.datetime_typed_label_check(threshold),
            )
        }
    )
    serialized = getattr(schema, f"to_{format}")()
    loaded = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)

    assert loaded.columns["ts"].checks[0].statistics == {
        "label": pd.Timestamp("2026-01-01 00:00:00")
    }
    frame = pd.DataFrame(
        {"ts": pd.Series(["2026-01-01"], dtype="datetime64[ns]")}
    )
    schema.validate(frame)
    loaded.validate(frame)


@pytest.mark.parametrize("format", ["json", "yaml"])
@pytest.mark.parametrize(
    "check_name, valid_offsets",
    [
        ("greater_than", {1}),
        ("greater_than_or_equal_to", {0, 1}),
        ("less_than", {-1}),
        ("less_than_or_equal_to", {-1, 0}),
        ("equal_to", {0}),
        ("not_equal_to", {-1, 1}),
        ("in_range", {0, 1}),
    ],
)
def test_datetime_comparison_registry_names(format, check_name, valid_offsets):
    """Every comparison uses its registry name despite a custom display name."""
    if format == "yaml":
        pytest.importorskip("yaml")
    threshold = pd.Timestamp("2026-01-01 00:00:00.500000001")
    args = (
        (threshold, threshold + pd.Timedelta(1, unit="ns"))
        if check_name == "in_range"
        else (threshold,)
    )
    check = getattr(pa.Check, check_name)(*args, name="custom display name")
    schema = pa.DataFrameSchema({"ts": pa.Column("datetime64[ns]", check)})
    serialized = getattr(schema, f"to_{format}")()
    loaded = getattr(pa.DataFrameSchema, f"from_{format}")(serialized)

    assert loaded.columns["ts"].checks[0].name == "custom display name"
    assert loaded.columns["ts"].checks[0].statistics == check.statistics
    for offset in [-1, 0, 1]:
        value = threshold + pd.Timedelta(offset, unit="ns")
        frame = pd.DataFrame(
            {"ts": pd.Series([value], dtype="datetime64[ns]")}
        )
        for validator in [schema, loaded]:
            if offset in valid_offsets:
                validator.validate(frame)
            else:
                with pytest.raises(pa.errors.SchemaError) as exc_info:
                    validator.validate(frame)
                assert (
                    exc_info.value.reason_code
                    == SchemaErrorReason.DATAFRAME_CHECK
                )
