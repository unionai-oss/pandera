#!/usr/bin/env bash
# Combine dorny/paths-filter outputs (see .github/filters.yaml) into the
# job-level flags and extras matrices consumed by ci-tests.yml.
#
# Usage:
#   PANDAS=true POLARS=false ... bash scripts/ci/derive_test_flags.sh
#
# Reads area booleans ("true"/"false") from environment variables, set in CI
# from the `changes` job's paths-filter step outputs. Prints KEY=VALUE flags
# to stdout (append to GITHUB_OUTPUT in CI).
#
# A change to shared/global config (the `global` filter), or to pandera/
# or tests/ code not covered by any backend-specific filter (the `common`
# filter), affects every area — see .github/filters.yaml for how those two
# filters are defined.

set -euo pipefail

is_true() { [[ "${1:-false}" == "true" ]]; }

common=false
if is_true "${GLOBAL:-}" || is_true "${COMMON:-}"; then
    common=true
fi

# combine <area-flag> -> "true" if the area-specific flag or `common` is true
combine() {
    if is_true "$1" || is_true "${common}"; then
        printf 'true'
    else
        printf 'false'
    fi
}

base=$(combine "${BASE:-}")
pandas=$(combine "${PANDAS:-}")
polars=$(combine "${POLARS:-}")
pyspark=$(combine "${PYSPARK:-}")
ibis=$(combine "${IBIS:-}")
geopandas=$(combine "${GEOPANDAS:-}")
dask=$(combine "${DASK:-}")
modin=$(combine "${MODIN:-}")
xarray=$(combine "${XARRAY:-}")
pyarrow=$(combine "${PYARROW:-}")
narwhals=$(combine "${NARWHALS:-}")
hypotheses=$(combine "${HYPOTHESES:-}")
io=$(combine "${IO:-}")
mypy=$(combine "${MYPY:-}")
strategies=$(combine "${STRATEGIES:-}")
fastapi=$(combine "${FASTAPI:-}")

import_test="${polars}"

supplemental=false
if is_true "$hypotheses" || is_true "$io" || is_true "$mypy" || is_true "$strategies" \
    || is_true "$fastapi" || is_true "$geopandas"; then
    supplemental=true
fi

dataframe=false
if is_true "$dask" || is_true "$polars" || is_true "$pyspark" || is_true "$modin" \
    || is_true "$ibis" || is_true "$xarray" || is_true "$pyarrow"; then
    dataframe=true
fi

narwhals_backend=false
if is_true "$polars" || is_true "$ibis" || is_true "$pyspark" || is_true "$narwhals"; then
    narwhals_backend=true
fi

json_array() {
    local out="["
    local first=1
    local item
    for item in "$@"; do
        if [[ $first -eq 1 ]]; then
            first=0
        else
            out+=","
        fi
        out+="\"${item}\""
    done
    printf '%s' "${out}]"
}

supplemental_extras=()
is_true "$hypotheses" && supplemental_extras+=("hypotheses")
is_true "$io" && supplemental_extras+=("io")
is_true "$mypy" && supplemental_extras+=("mypy")
is_true "$strategies" && supplemental_extras+=("strategies")
is_true "$fastapi" && supplemental_extras+=("fastapi")
is_true "$geopandas" && supplemental_extras+=("geopandas")

dataframe_extras=()
is_true "$dask" && dataframe_extras+=("dask")
is_true "$polars" && dataframe_extras+=("polars")
is_true "$pyspark" && dataframe_extras+=("pyspark")
if is_true "$modin"; then
    dataframe_extras+=("modin-dask" "modin-ray")
fi
is_true "$ibis" && dataframe_extras+=("ibis")
is_true "$xarray" && dataframe_extras+=("xarray")
is_true "$pyarrow" && dataframe_extras+=("pyarrow")

narwhals_backend_extras=()
if is_true "$narwhals"; then
    # The narwhals backend is shared code; exercise it for every library.
    narwhals_backend_extras+=("polars" "ibis" "pyspark")
else
    is_true "$polars" && narwhals_backend_extras+=("polars")
    is_true "$ibis" && narwhals_backend_extras+=("ibis")
    is_true "$pyspark" && narwhals_backend_extras+=("pyspark")
fi

# Human-readable summary for CI logs.
{
    echo "common=$common"
    echo "base=$base pandas=$pandas supplemental=$supplemental dataframe=$dataframe"
    echo "narwhals=$narwhals narwhals_backend=$narwhals_backend"
} >&2

# Machine-readable flags.
printf 'import_test=%s\n' "${import_test}"
printf 'base=%s\n' "${base}"
printf 'pandas=%s\n' "${pandas}"
printf 'supplemental=%s\n' "${supplemental}"
printf 'supplemental_extras=%s\n' "$(json_array "${supplemental_extras[@]+"${supplemental_extras[@]}"}")"
printf 'dataframe=%s\n' "${dataframe}"
printf 'dataframe_extras=%s\n' "$(json_array "${dataframe_extras[@]+"${dataframe_extras[@]}"}")"
printf 'narwhals_backend=%s\n' "${narwhals_backend}"
printf 'narwhals_backend_extras=%s\n' "$(json_array "${narwhals_backend_extras[@]+"${narwhals_backend_extras[@]}"}")"
printf 'narwhals=%s\n' "${narwhals}"
