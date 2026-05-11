# Phase 2 Triage Report

**Date:** 2026-05-11
**Command:** `JAVA_HOME=/Users/deepyaman/Library/Caches/rattler/cache/pkgs/openjdk-17.0.17-h99a4030_0/lib/jvm PANDERA_USE_NARWHALS_BACKEND=True python -m pytest tests/pyspark/ --ignore=tests/pyspark/test_schemas_on_pyspark_pandas.py -k "spark and not spark_connect" -v --no-header -q --tb=short`
**Outcome summary:** 59 passed, 27 xfailed (passing), 300 failed (Category C — PHASE SPLIT RECOMMENDED), 1 xpass-strict (Category A correction), 0 Category B, 360 deselected (spark_connect)

## Test Results

| File | Test ID | Outcome | Category | Disposition | Reason or Fix Reference |
|---|---|---|---|---|---|
| test_pyspark_accessor.py | test_modin_accessor_warning | PASSED | — | pass | Accessor warning works regardless of backend |
| test_pyspark_accessor.py | test_dataframe_add_schema[...] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestCustomCheck::test_extension[spark] | XFAIL | A | already xfail (Plan 01) | Custom checks use PysparkDataframeColumnObject API |
| test_pyspark_check.py | TestCustomCheck::test_extension_dataframe_model[spark] | XFAIL | A | already xfail (Plan 01) | Custom checks use PysparkDataframeColumnObject API |
| test_pyspark_check.py | TestEqualToCheck::test_equal_to_check[data0..data10-spark] (22 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestEqualToCheck::test_failed_unaccepted_datatypes[data0..data1-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestNotEqualToCheck::test_not_equal_to_check[data0..data10-spark] (22 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestNotEqualToCheck::test_failed_unaccepted_datatypes[data0..data1-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestGreaterThanCheck::test_greater_than_check[data0..data8-spark] (18 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestGreaterThanCheck::test_failed_unaccepted_datatypes[data2..data3-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestGreaterThanCheck::test_failed_unaccepted_datatypes[data0..data1-spark] (4 tests) | PASSED | — | pass | TypeError raised pre-validation |
| test_pyspark_check.py | TestGreaterThanEqualToCheck::test_greater_than_or_equal_to_check[data0..data8-spark] (18 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestGreaterThanEqualToCheck::test_failed_unaccepted_datatypes[data0..data1-spark] (4 tests) | PASSED | — | pass | TypeError raised pre-validation |
| test_pyspark_check.py | TestLessThanCheck::test_less_than_check[data0..data8-spark] (18 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestLessThanCheck::test_failed_unaccepted_datatypes[data0..data3-spark] (8 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestLessThanCheck::test_failed_none_expression[data0-spark] (2 tests) | PASSED | — | pass | Check construction raises ValueError pre-validation |
| test_pyspark_check.py | TestLessThanOrEqualToCheck::test_less_than_or_equal_to_check[data0..data8-spark] (18 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestLessThanOrEqualToCheck::test_failed_unaccepted_datatypes[data0..data3-spark] (8 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestLessThanOrEqualToCheck::test_failed_none_expression[data0-spark] (2 tests) | PASSED | — | pass | Check construction raises ValueError pre-validation |
| test_pyspark_check.py | TestIsInCheck::test_isin_check[data0..data4-spark] (10 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestIsInCheck::test_failed_unaccepted_datatypes[data0-spark] | PASSED | — | pass | TypeError raised pre-validation |
| test_pyspark_check.py | TestNotInCheck::test_notin_check[data0..data4-spark] (10 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestNotInCheck::test_failed_unaccepted_datatypes[data0-spark] | PASSED | — | pass | TypeError raised pre-validation |
| test_pyspark_check.py | TestInRangeCheck::test_inrange_include_min_max_check[data0..data8-spark] (9 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestInRangeCheck::test_inrange_exclude_min_only_check[data0..data8-spark] (9 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestInRangeCheck::test_inrange_exclude_max_only_check[data0..data8-spark] (9 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestInRangeCheck::test_inrange_exclude_min_max_check[data0..data8-spark] (9 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestInRangeCheck::test_failed_unaccepted_datatypes[data0..data3-spark] (4 tests) | PASSED | — | pass | TypeError raised pre-validation |
| test_pyspark_check.py | TestDecorator::test_datatype_check_decorator[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals backend raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_check.py | TestUniqueValuesEqCheck::test_unique_values_eq_check[data0..data9-spark] (10 tests) | XFAIL | A | already xfail (Plan 01) | unique_values_eq not registered for Narwhals backend |
| test_pyspark_check.py | TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data0-datatype0-spark] | XPASS(strict) | A-FIX | remove xfail for BooleanType param — TypeError still raised under narwhals | XPASS: xfail was too broad; BooleanType TypeError raised pre-validation |
| test_pyspark_check.py | TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data1-datatype1-spark] | XFAIL | A | already xfail (Plan 01) — ArrayType, TypeError not raised under narwhals | unique_values_eq not registered for Narwhals backend |
| test_pyspark_check.py | TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data2-datatype2-spark] | XFAIL | A | already xfail (Plan 01) — MapType, TypeError not raised under narwhals | unique_values_eq not registered for Narwhals backend |
| test_pyspark_config.py | TestPanderaConfig::test_disable_validation[spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_schema_only[spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_data_only[spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_schema_and_data[spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_cache_dataframe_settings[True-True-spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_cache_dataframe_settings[True-False-spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_cache_dataframe_settings[False-True-spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_config.py | TestPanderaConfig::test_cache_dataframe_settings[False-False-spark] | XFAIL | A | already xfail (Plan 01) | Config dict assertions hardcode use_narwhals_backend=False |
| test_pyspark_container.py | test_pyspark_sample[spark] | XFAIL | A | already xfail (Plan 01) | sample= not supported in Narwhals backend |
| test_pyspark_container.py | test_pyspark_dataframeschema[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors; dtype string mismatch (Int64 vs LongType()) |
| test_pyspark_container.py | test_pyspark_dataframeschema_with_alias_types[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors; dtype string mismatch |
| test_pyspark_container.py | test_pyspark_regex_column[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_container.py | test_pyspark_nullable[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_container.py | test_pyspark_unique_config[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_container.py | test_pyspark_column_metadata[spark] | PASSED | — | pass | Metadata test does not validate data |
| test_pyspark_container.py | test_pyspark_unique_field[spark] | PASSED | — | pass | Does not trigger data validation |
| test_pyspark_container.py | test_schema_to_structtype[spark] | PASSED | — | pass | Schema serialization only |
| test_pyspark_container.py | test_schema_to_ddl[spark] | PASSED | — | pass | Schema serialization only |
| test_pyspark_container.py | test_pyspark_read[spark] | PASSED | — | pass | Schema reading only |
| test_pyspark_decorators.py | TestPanderaDecorators::test_cache_dataframe_requirements[spark] | PASSED | — | pass | FakeDataFrameSchemaBackend is backend-agnostic (A1 confirmed) |
| test_pyspark_decorators.py | TestPanderaDecorators::test_cache_dataframe_settings[True-True-True-None-spark] | XFAIL | A | already xfail (Plan 01) | Narwhals bypasses PySpark caching decorators |
| test_pyspark_decorators.py | TestPanderaDecorators::test_cache_dataframe_settings[True-False-True-True-spark] | XFAIL | A | already xfail (Plan 01) | Narwhals bypasses PySpark caching decorators |
| test_pyspark_decorators.py | TestPanderaDecorators::test_cache_dataframe_settings[False-True-None-None-spark] | XFAIL | A | already xfail (Plan 01) | Narwhals bypasses PySpark caching decorators |
| test_pyspark_decorators.py | TestPanderaDecorators::test_cache_dataframe_settings[False-False-None-None-spark] | XFAIL | A | already xfail (Plan 01) | Narwhals bypasses PySpark caching decorators |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_float_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals materializes data causing STRUCT_ARRAY_LENGTH_MISMATCH; also dtype mismatch |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_double_types[data0..data3-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_decimal_default_types[data0..data3-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_decimal_parameterized_types[data0-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_int_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_longint_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_shortint_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllNumericTypes::test_pyspark_all_bytetint_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllDatetimeTestClass::test_pyspark_all_date_types[data0..data3-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestAllDatetimeTestClass::test_pyspark_all_datetime_types[data0..data8-spark] (9 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestBinaryStringTypes::test_pyspark_all_binary_types[data0..data3-spark] (4 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestBinaryStringTypes::test_pyspark_all_string_types[data0..data4-spark] (5 tests) | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestComplexType::test_pyspark_array_type[data0-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_dtypes.py | TestComplexType::test_pyspark_map_type[data0-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Same root cause |
| test_pyspark_engine.py | test_pyspark_data_type[Bool/Double/MapType/ShortInt/...] (13 tests) | PASSED | — | pass | Engine dtype tests are backend-agnostic |
| test_pyspark_error.py | test_dataframe_add_schema[spark] (not run — no spark param) | N/A | — | no spark fixture | Test uses inline data without spark fixture |
| test_pyspark_error.py | test_pyspark_check_eq[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors + _concat_failure_cases crashes with _df AttributeError |
| test_pyspark_error.py | test_pyspark_check_nullable[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_error.py | test_pyspark_schema_data_checks[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_error.py | test_pyspark_fields[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_error.py | test_pyspark__error_handler_lazy_validation[spark] | PASSED | — | pass | Only checks SchemaErrors is raised (not accessor pattern) |
| test_pyspark_model.py | test_schema_with_bare_types[spark] | PASSED | — | pass | Schema definition only |
| test_pyspark_model.py | test_schema_with_bare_types_and_field[spark] | PASSED | — | pass | Schema definition only |
| test_pyspark_model.py | test_pyspark_fields_metadata[spark] | PASSED | — | pass | Metadata only |
| test_pyspark_model.py | test_docstring_substitution[spark] | PASSED | — | pass | Docstring check only |
| test_pyspark_model.py | test_optional_column[spark] | PASSED | — | pass | No data validation |
| test_pyspark_model.py | test_invalid_field[spark] | PASSED | — | pass | Schema definition error test |
| test_pyspark_model.py | test_schema_to_structtype[spark] | PASSED | — | pass | Schema serialization only |
| test_pyspark_model.py | test_schema_to_ddl[spark] | PASSED | — | pass | Schema serialization only |
| test_pyspark_model.py | test_inherited_schema_to_structtype[spark] | PASSED | — | pass | Schema serialization only |
| test_pyspark_model.py | test_schema_with_bare_types_field_and_checks[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_model.py | test_schema_with_bare_types_field_type[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors + dtype mismatch |
| test_pyspark_model.py | test_pyspark_bare_fields[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_model.py | test_dataframe_schema_unique[no_data-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors + dtype mismatch (IntegerType() vs Int32) |
| test_pyspark_model.py | test_dataframe_schema_unique[unique_data-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors + dtype mismatch |
| test_pyspark_model.py | test_dataframe_schema_unique[duplicated_data-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors + dtype mismatch |
| test_pyspark_model.py | test_dataframe_schema_unique_wrong_column[wrong_column-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors |
| test_pyspark_model.py | test_dataframe_schema_unique_wrong_column[multiple_wrong_columns-spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors |
| test_pyspark_model.py | test_dataframe_schema_strict[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_model.py | test_validation_succeeds_with_missing_optional_column[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_model.py | test_registered_dataframemodel_checks[spark] | FAILED | C | [DEFERRED — exceeded per-run cap] | Narwhals raises SchemaErrors instead of setting df.pandera.errors |
| test_pyspark_sql_io.py | test_pyspark_yaml_json_roundtrip | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | test_pyspark_full_serdes_includes_version | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | test_pyspark_rejects_polars_schema_type | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | TestPySparkDataFrameModelIO::test_model_to_yaml | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | TestPySparkDataFrameModelIO::test_model_to_json | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | TestPySparkDataFrameModelIO::test_model_from_yaml | PASSED | — | pass | Serialization only |
| test_pyspark_sql_io.py | TestPySparkDataFrameModelIO::test_model_from_json | PASSED | — | pass | Serialization only |
| test_schema_inference.py | test_infer_dataframe_schema | PASSED | — | pass | Uses mock Spark, no real narwhals execution |
| test_schema_inference.py | test_infer_schema_alias | PASSED | — | pass | Uses mock Spark, no real narwhals execution |
| test_schema_inference.py | test_infer_dataframe_schema_wrong_type | PASSED | — | pass | Uses mock Spark, no real narwhals execution |

## Category B — New xfail markers to add (handled in Task 2)

One XPASS (strict) correction required — existing xfail is too broad.

### XPASS correction: TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data0-datatype0]

The method-level xfail on `test_failed_unaccepted_datatypes` covers 3 parametrizations (BooleanType, ArrayType, MapType). BooleanType (data0) PASSES under narwhals because the `unique_values_eq` check raises `TypeError` for Boolean even before narwhals gets involved. The xfail with `strict=True` causes this to fail as XPASS(strict).

Fix: Restructure to apply per-parametrization xfail marks for ArrayType (data1) and MapType (data2) only; remove method-level xfail.

- File: tests/pyspark/test_pyspark_check.py
- Test: TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data0-datatype0-spark]
- Failure summary: `[XPASS(strict)] unique_values_eq not registered for Narwhals backend`
- Failure output:
  ```
  FAILED tests/pyspark/test_pyspark_check.py::TestUniqueValuesEqCheck::test_failed_unaccepted_datatypes[data0-datatype0-spark] - Failed: [XPASS(strict)] unique_values_eq not registered for Narwhals backend
  ```
- Proposed resolution: Remove method-level xfail; add inline pytest.param marks for data1/ArrayType and data2/MapType in get_data_param(); update pytest_generate_tests() to use pytest.param with marks.

## Category C — Backend bugs to fix (handled in Task 3)

**3 root cause bugs found. All 3 exceed the per-run cap of 3 fixes in combination because root cause 1 requires >20 lines of core backend changes. PHASE SPLIT RECOMMENDED.**

### Category C Entry 1 (PRIMARY): Narwhals backend raises SchemaErrors instead of setting df.pandera.errors

- Test: All ~290 tests across test_pyspark_check.py, test_pyspark_container.py, test_pyspark_model.py, test_pyspark_error.py, test_pyspark_accessor.py that invoke `schema.validate(df)` and then check `df_out.pandera.errors`
- Failure summary: `pandera.errors.SchemaErrors: {...}` raised instead of `check_obj.pandera.errors = error_dicts; return check_obj`
- Failure output (representative — test_pyspark_check.py::TestEqualToCheck::test_equal_to_check[data0-datatype0-equal_to-spark]):
  ```
  FAILED tests/pyspark/test_pyspark_check.py::TestEqualToCheck::test_equal_to_check[data0-datatype0-equal_to-spark]
  pandera/backends/narwhals/container.py:227: SchemaErrors
  E               pandera.errors.SchemaErrors: {
  E                   "DATA": {
  E                       "DATAFRAME_CHECK": [
  E                           {
  E                               "schema": null,
  E                               "column": "code",
  E                               "check": "equal_to(30)",
  E                               "error": "Check '<Check equal_to: equal_to(30)>' failed."
  E                           }
  E                       ]
  E                   }
  E               }
  ```
- Likely source: pandera/backends/narwhals/container.py lines 219-233 (raises SchemaErrors instead of setting df.pandera.errors)
- Fix approach: In DataFrameSchemaBackend.validate(), when the input is a PySpark DataFrame (detectable via `_is_sql_lazy(check_lf)`), adapt error handling to set `check_obj.pandera.errors = error_dicts` and return `check_obj` instead of raising SchemaErrors. Requires: (a) keeping the native check_obj reference, (b) building error_dicts via error_handler.summarize() like the native PySpark backend, (c) setting accessor errors on the native df. Estimated: 30-50 lines change in container.py + base.py. [DEFERRED — exceeded per-run cap]

### Category C Entry 2: _concat_failure_cases crashes with AttributeError when PySpark DataFrames are in failure case collection

- Test: test_pyspark_error.py::test_pyspark_check_eq[spark]
- Failure summary: `PySparkAttributeError: [ATTRIBUTE_NOT_SUPPORTED] Attribute '_df' is not supported`
- Failure output:
  ```
  FAILED tests/pyspark/test_pyspark_error.py::test_pyspark_check_eq[spark] - pyspark.errors.exceptions.base.PySparkAttributeError: [ATTRIBUTE_NOT_SUPPORTED] Attribute `_df` is not supported.
  pandera/backends/narwhals/base.py:53: PySparkAttributeError
  ```
  Full traceback: `pl.concat(elems)` receives items containing a native PySpark DataFrame (from `_build_lazy_failure_case` → `nw.to_native(enriched)`) mixed with a `pl.DataFrame` (from `_build_scalar_failure_case`). Polars treats the first item as `pl.DataFrame` (isinstance check passes for data0), then calls `plr.concat_df(elems)` which internally accesses `._df` on the PySpark DataFrame.
- Likely source: pandera/backends/narwhals/base.py:41-53 (_concat_failure_cases function)
- Fix approach: Add a PySpark DataFrame detection guard in `_concat_failure_cases` — check `type(first).__module__.startswith('pyspark')` and use PySpark `.union()` for PySpark frames, similar to the ibis path. [DEFERRED — exceeded per-run cap]

### Category C Entry 3: Narwhals dtype string format differs from PySpark

- Test: test_pyspark_container.py::test_pyspark_dataframeschema[spark], test_pyspark_model.py::test_dataframe_schema_unique[*]
- Failure summary: `expected column 'age' to have type IntegerType(), got Int64` — narwhals reports `Int64` for a LongType column
- Failure output:
  ```
  FAILED tests/pyspark/test_pyspark_container.py::test_pyspark_dataframeschema[spark]
  E   pandera.errors.SchemaErrors: {
  E       "SCHEMA": {
  E           "WRONG_DATATYPE": [
  E               {
  E                   "schema": null,
  E                   "column": "age",
  E                   "check": "dtype('IntegerType()')",
  E                   "error": "expected column 'age' to have type IntegerType(), got Int64"
  E               }
  E           ]
  E       }
  E   }
  
  FAILED tests/pyspark/test_pyspark_model.py::test_dataframe_schema_unique[no_data-spark]
  E   pandera.errors.SchemaErrors: {
  E       "SCHEMA": {
  E           "WRONG_DATATYPE": [
  E               {
  E                   "schema": "UniqueSingleColumn",
  E                   "column": "a",
  E                   "check": "dtype('IntegerType()')",
  E                   "error": "expected column 'a' to have type IntegerType(), got Int32"
  E               }
  E           ]
  E       }
  E   }
  ```
- Likely source: pandera/backends/narwhals/components.py:268 (`f"got {nw_dtype}"` where nw_dtype is narwhals dtype string, not PySpark dtype string)
- Fix approach: In `ColumnBackend.check_dtype()`, use the native PySpark dtype string (from `nw.to_native(frame).schema[column]`) instead of `str(nw_dtype)` for PySpark frames. [DEFERRED — exceeded per-run cap]

### Category C Entry 4: Narwhals materializes PySpark data during validation (STRUCT_ARRAY_LENGTH_MISMATCH)

- Test: All test_pyspark_dtypes.py tests that use spark_df() with sample_data (2-column tuples, 1-column schema, verifySchema=False)
- Failure summary: `IllegalArgumentException: [STRUCT_ARRAY_LENGTH_MISMATCH] Input row doesn't have expected number of values required by the schema. 1 fields are required while 2 values are provided.`
- Failure output:
  ```
  FAILED tests/pyspark/test_pyspark_dtypes.py::TestAllNumericTypes::test_pyspark_all_int_types[int0-spark]
  E   pyspark.errors.exceptions.captured.IllegalArgumentException: [STRUCT_ARRAY_LENGTH_MISMATCH] Input row doesn't have expected number of values required by the schema. 1 fields are required while 2 values are provided. SQLSTATE: 2201E
  ```
- Likely source: pandera/backends/narwhals/base.py:124 (`_materialize(passed_lf)`) — this collects the PySpark DataFrame, triggering Spark to actually read the data and discover the schema mismatch that verifySchema=False suppressed.
- Fix approach: The native PySpark backend never materializes/collects during validation — it only reads the schema. The narwhals backend must avoid materializing PySpark DataFrames in run_check(). However, the check result (pass/fail) requires executing the check expression. Resolution: use a lazy-only execution mode for PySpark (no collect during check, only during error reporting). [DEFERRED — exceeded per-run cap]

**PHASE SPLIT RECOMMENDED:** All 4 Category C entries share root cause: the narwhals backend was designed for polars/ibis which have different API contracts than the native PySpark backend. A follow-on Plan 04 should be created to rearchitect the narwhals DataFrameSchemaBackend.validate() for PySpark, making it:
1. Set `check_obj.pandera.errors` instead of raising `SchemaErrors`
2. Handle dtype string format using PySpark-native strings
3. Fix `_concat_failure_cases` for PySpark DataFrames
4. Avoid materializing PySpark DataFrames unnecessarily

## Category D — Pre-existing failures (no action)

- spark_connect fixture errors: ~360 tests filtered by `-k "not spark_connect"` (no live Spark Connect server)
- Others: None

## Resolved Assumptions (from RESEARCH.md)

- A1 (test_cache_dataframe_requirements passes): confirmed — test_pyspark_decorators.py::TestPanderaDecorators::test_cache_dataframe_requirements[spark] PASSED
- A2 (test_cache_dataframe_settings fails): confirmed — all 4 parametrizations XFAIL as expected
- A3 (test_pyspark_dtypes.py mostly passes): refuted — all 58 tests FAILED (narwhals materializes PySpark data causing IllegalArgumentException + dtype string mismatch)
- A4 (test_pyspark_error.py mostly passes — dtype string risk): refuted — 4 of 5 tests FAILED; the failures are more severe than anticipated (SchemaErrors raised + _concat_failure_cases crash with _df AttributeError)
- A5 (spark_connect failures are pre-existing): confirmed — spark_connect tests filtered by -k, no narwhals-specific issues there
- A6 (pyspark pulls pandas transitively): confirmed — no ImportError for pandas in the test run
- A7 (TestCustomCheck tests fail because NarwhalsData doesn't have .column_name/.dataframe): confirmed — both TestCustomCheck tests XFAIL as expected

## SC2c Decision

- **Decision:** scope-out
- **Date:** 2026-05-11
- **Reference:** 02-VERIFICATION.md gap entry (Phase 2 SC2c human verification item)

ROADMAP Phase 2 SC2 originally required that "row-index in `failure_cases`" be covered by at
least one xfail-marked PySpark test. This clause was written in the context of polars/ibis, where
integer row indices either exist (polars) or existed before the narwhals backend removed them
(ibis — hence the existing xfail tests in `tests/ibis/test_ibis_container.py`).

PySpark DataFrames are distributed and partitioned with no native integer row index. There is no
`with_row_index` equivalent that produces a stable ordering across executors. Creating an xfail
test asserting the absence of a row index would document a non-feature that can never be provided
by any future narwhals version — adding noise without value.

The ibis xfail pattern (e.g. `test_failed_cases_index_for_column_check`) xfails pre-existing tests
where row-index *was* available in the native ibis backend and the narwhals backend removed it.
PySpark never had this feature in its native backend either. Scope-out is the consistent choice.

ROADMAP SC2 has been updated to: "Element-wise checks and `sample=`/`tail=` params are each
covered by at least one `xfail`-marked test. Row-index in `failure_cases` is inapplicable to
PySpark — PySpark DataFrames are distributed and partitioned with no native integer row index;
this clause applies only to polars/ibis backends."
