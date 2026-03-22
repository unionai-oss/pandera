---
created: 2026-03-22T02:16:13.061Z
title: Push synthetic column construction into schema API layer
area: general
files:
  - pandera/backends/narwhals/container.py:318-323
---

## Problem

In `collect_schema_components`, when `schema.columns` is empty but `schema.dtype` is set, the narwhals backend synthetically creates Column objects for each frame column. This requires the backend to know the framework-specific Column class (polars vs ibis), which is an abstraction leak — a framework-agnostic backend shouldn't reach into `pandera.api.polars.components` or `pandera.api.ibis.components`.

Currently mitigated with `importlib` dynamic import (keyed off `schema.__class__.__module__`), but the root fix is to push this logic into the schema API layer.

## Solution

Add a method to the DataFrameSchema API (e.g., `schema.infer_columns(frame_column_names)`) that returns the appropriate Column instances. The backend calls this method instead of importing a framework-specific Column class directly. The schema knows its own Column type; the backend shouldn't need to.
