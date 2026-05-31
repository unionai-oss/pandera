# Deferred Items — Phase 04

## Pre-existing failures (out of scope for Phase 04-03)

### test_custom_check_receives_table_and_key
- **File:** tests/backends/narwhals/test_e2e.py::TestCustomChecksIbis::test_custom_check_receives_table_and_key
- **Issue:** Test asserts `table_type == "DatabaseTable"` but ibis changed the class name to `"Table"` in a newer version.
- **Confirmed pre-existing:** Fails on `e3dcb10` (before Task 2 changes) and on `5b69446` (before Task 1 changes).
- **Action needed:** Update test to accept either `"DatabaseTable"` or `"Table"` as the ibis Table class name.
