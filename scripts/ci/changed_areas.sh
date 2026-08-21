#!/usr/bin/env bash
# Classify files changed in a PR/push into pandera CI areas.
#
# Usage:
#   bash scripts/ci/changed_areas.sh <file> [<file> ...]
#   git diff --name-only base head | bash scripts/ci/changed_areas.sh
#
# Prints KEY=VALUE flags to stdout (append to GITHUB_OUTPUT in CI), e.g.:
#   lint=true
#   ibis=true
#   dataframe_extras=["ibis"]
#
# Rules:
#   - A file under a backend-specific path (e.g. pandera/api/ibis/, tests/ibis/)
#     triggers only that backend's test area. A file may belong to several
#     areas (e.g. pandera/io/ibis_io.py triggers both `ibis` and `io`).
#   - Any other change under pandera/ or tests/, or to shared build config
#     (pyproject.toml, requirements, .github/ ...), is "common" and triggers
#     every unit-test area, since shared code can affect any backend.
#   - Pure docs changes trigger only the docs job.
#
# Area flags are final: `area=true` means (common || area-specific change).

set -euo pipefail

# ---------------------------------------------------------------------------
# Collect changed files: from arguments, or from stdin when no arguments.
# ---------------------------------------------------------------------------
files=()
if [[ $# -gt 0 ]]; then
    files=("$@")
else
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            files+=("$line")
        fi
    done
fi

# ---------------------------------------------------------------------------
# Path matching helpers. In bash pattern matching `*` spans `/`, so
# `pandera/api/ibis/*` also matches files in nested subdirectories.
# ---------------------------------------------------------------------------
matches_any() {
    local f="$1"
    shift
    local pat
    for pat in "$@"; do
        # shellcheck disable=SC2254  # intentional pattern matching
        case "$f" in
        $pat)
            return 0
            ;;
        esac
    done
    return 1
}

area_pandas() {
    matches_any "$1" \
        'pandera/api/pandas/*' \
        'pandera/backends/pandas/*' \
        'pandera/engines/pandas_engine.py' \
        'pandera/engines/numpy_engine.py' \
        'pandera/pandas.py' \
        'pandera/_pandas_deprecated.py' \
        'pandera/_patch_numpy2.py' \
        'pandera/accessors/*' \
        'pandera/io/pandas_io.py' \
        'pandera/typing/pandas.py' \
        'tests/pandas/*'
}

area_polars() {
    matches_any "$1" \
        'pandera/api/polars/*' \
        'pandera/backends/polars/*' \
        'pandera/engines/polars_engine.py' \
        'pandera/polars.py' \
        'pandera/io/polars_io.py' \
        'pandera/typing/polars.py' \
        'tests/polars/*' \
        'tests/common/*'
}

area_pyspark() {
    # Note: tests/cli/ and the CLI modules also run in the pyspark nox session
    # (PANDERA_RUN_SPARK_CLI=1), so they belong to this area as well.
    matches_any "$1" \
        'pandera/api/pyspark/*' \
        'pandera/backends/pyspark/*' \
        'pandera/engines/pyspark_engine.py' \
        'pandera/pyspark.py' \
        'pandera/io/pyspark_sql_io.py' \
        'pandera/typing/pyspark.py' \
        'pandera/typing/pyspark_sql.py' \
        'tests/pyspark/*' \
        'pandera/cli.py' \
        'pandera/_cli/*' \
        'tests/cli/*'
}

area_ibis() {
    matches_any "$1" \
        'pandera/api/ibis/*' \
        'pandera/backends/ibis/*' \
        'pandera/engines/ibis_engine.py' \
        'pandera/ibis.py' \
        'pandera/io/ibis_io.py' \
        'pandera/typing/ibis.py' \
        'tests/ibis/*' \
        'tests/common/*'
}

area_geopandas() {
    matches_any "$1" \
        'pandera/api/geopandas/*' \
        'pandera/engines/geopandas_engine.py' \
        'pandera/geopandas.py' \
        'pandera/typing/geopandas.py' \
        'tests/geopandas/*'
}

area_dask() {
    matches_any "$1" 'tests/dask/*' 'pandera/typing/dask.py'
}

area_modin() {
    matches_any "$1" 'tests/modin/*' 'pandera/typing/modin.py'
}

area_xarray() {
    matches_any "$1" \
        'pandera/api/xarray/*' \
        'pandera/backends/xarray/*' \
        'pandera/engines/xarray_engine.py' \
        'pandera/xarray.py' \
        'pandera/io/xarray_io.py' \
        'pandera/typing/xarray.py' \
        'tests/xarray/*'
}

area_pyarrow() {
    matches_any "$1" \
        'pandera/api/pyarrow/*' \
        'pandera/backends/pyarrow/*' \
        'pandera/engines/pyarrow_engine.py' \
        'pandera/pyarrow.py' \
        'pandera/typing/pyarrow.py' \
        'tests/pyarrow/*'
}

area_narwhals() {
    matches_any "$1" \
        'pandera/api/narwhals/*' \
        'pandera/backends/narwhals/*' \
        'pandera/engines/narwhals_engine.py' \
        'tests/narwhals/*'
}

area_hypotheses() {
    matches_any "$1" \
        'pandera/api/hypotheses.py' \
        'pandera/schema_statistics/*' \
        'tests/hypotheses/*'
}

area_io() {
    matches_any "$1" 'pandera/io/*' 'tests/io/*'
}

area_mypy() {
    matches_any "$1" \
        'pandera/typing/*' \
        'pandera/mypy.py' \
        'tests/mypy/*' \
        'tests/pyright/*'
}

area_strategies() {
    matches_any "$1" 'pandera/strategies/*' 'tests/strategies/*'
}

area_fastapi() {
    matches_any "$1" 'tests/fastapi/*' 'pandera/typing/fastapi.py'
}

area_base() {
    # The extra=None session runs tests/base/ and tests/cli/
    matches_any "$1" \
        'tests/base/*' \
        'tests/cli/*' \
        'tests/conftest.py' \
        'tests/__init__.py' \
        'pandera/cli.py' \
        'pandera/_cli/*' \
        'pandera/__main__.py'
}

# Shared build/CI config: changes here can affect any area.
is_global() {
    matches_any "$1" \
        'pyproject.toml' \
        'setup.py' \
        'noxfile.py' \
        'requirements.txt' \
        'requirements-*.txt' \
        'environment.yml' \
        '.coveragerc' \
        '.github/*' \
        'scripts/*'
}

# ---------------------------------------------------------------------------
# Classify each changed file.
# ---------------------------------------------------------------------------
common=0
lint=0
docs=0
pandas=0
polars=0
pyspark=0
ibis=0
geopandas=0
dask=0
modin=0
xarray=0
pyarrow=0
narwhals=0
hypotheses=0
io=0
mypy=0
strategies=0
fastapi=0
base=0

for f in ${files[@]+"${files[@]}"}; do
    specific=0
    if area_pandas "$f"; then pandas=1; specific=1; fi
    if area_polars "$f"; then polars=1; specific=1; fi
    if area_pyspark "$f"; then pyspark=1; specific=1; fi
    if area_ibis "$f"; then ibis=1; specific=1; fi
    if area_geopandas "$f"; then geopandas=1; specific=1; fi
    if area_dask "$f"; then dask=1; specific=1; fi
    if area_modin "$f"; then modin=1; specific=1; fi
    if area_xarray "$f"; then xarray=1; specific=1; fi
    if area_pyarrow "$f"; then pyarrow=1; specific=1; fi
    if area_narwhals "$f"; then narwhals=1; specific=1; fi
    if area_hypotheses "$f"; then hypotheses=1; specific=1; fi
    if area_io "$f"; then io=1; specific=1; fi
    if area_mypy "$f"; then mypy=1; specific=1; fi
    if area_strategies "$f"; then strategies=1; specific=1; fi
    if area_fastapi "$f"; then fastapi=1; specific=1; fi
    if area_base "$f"; then base=1; specific=1; fi

    if is_global "$f"; then
        common=1
    elif [[ "$f" == pandera/* || "$f" == tests/* ]] && [[ $specific -eq 0 ]]; then
        # Shared code not assigned to any backend-specific area.
        common=1
    fi

    # Lint runs on any code or tooling-config change.
    if is_global "$f"; then
        lint=1
    fi
    case "$f" in
    *.py | *.pyi | pandera/* | tests/* | mypy.ini | .pylintrc) lint=1 ;;
    esac

    # Docs builds run doctests over the package source.
    if is_global "$f"; then
        docs=1
    fi
    case "$f" in
    docs/* | .readthedocs.yml | Makefile) docs=1 ;;
    esac
done

if [[ $common -eq 1 ]]; then
    lint=1
    docs=1
fi

# ---------------------------------------------------------------------------
# Derive final area flags (common || area-specific) and job-level flags.
# ---------------------------------------------------------------------------
base=$(( common + base > 0 ))
pandas=$(( common + pandas > 0 ))
polars=$(( common + polars > 0 ))
pyspark=$(( common + pyspark > 0 ))
ibis=$(( common + ibis > 0 ))
geopandas=$(( common + geopandas > 0 ))
dask=$(( common + dask > 0 ))
modin=$(( common + modin > 0 ))
xarray=$(( common + xarray > 0 ))
pyarrow=$(( common + pyarrow > 0 ))
narwhals=$(( common + narwhals > 0 ))
hypotheses=$(( common + hypotheses > 0 ))
io=$(( common + io > 0 ))
mypy=$(( common + mypy > 0 ))
strategies=$(( common + strategies > 0 ))
fastapi=$(( common + fastapi > 0 ))

import_test=$polars
supplemental=$(( hypotheses + io + mypy + strategies + fastapi + geopandas > 0 ))
dataframe=$(( dask + polars + pyspark + modin + ibis + xarray + pyarrow > 0 ))
narwhals_backend=$(( polars + ibis + pyspark + narwhals > 0 ))

boolify() {
    if [[ "$1" -eq 1 ]]; then
        printf 'true'
    else
        printf 'false'
    fi
}

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
if [[ $hypotheses -eq 1 ]]; then supplemental_extras+=("hypotheses"); fi
if [[ $io -eq 1 ]]; then supplemental_extras+=("io"); fi
if [[ $mypy -eq 1 ]]; then supplemental_extras+=("mypy"); fi
if [[ $strategies -eq 1 ]]; then supplemental_extras+=("strategies"); fi
if [[ $fastapi -eq 1 ]]; then supplemental_extras+=("fastapi"); fi
if [[ $geopandas -eq 1 ]]; then supplemental_extras+=("geopandas"); fi

dataframe_extras=()
if [[ $dask -eq 1 ]]; then dataframe_extras+=("dask"); fi
if [[ $polars -eq 1 ]]; then dataframe_extras+=("polars"); fi
if [[ $pyspark -eq 1 ]]; then dataframe_extras+=("pyspark"); fi
if [[ $modin -eq 1 ]]; then
    dataframe_extras+=("modin-dask")
    dataframe_extras+=("modin-ray")
fi
if [[ $ibis -eq 1 ]]; then dataframe_extras+=("ibis"); fi
if [[ $xarray -eq 1 ]]; then dataframe_extras+=("xarray"); fi
if [[ $pyarrow -eq 1 ]]; then dataframe_extras+=("pyarrow"); fi

narwhals_backend_extras=()
if [[ $narwhals -eq 1 ]]; then
    # The narwhals backend is shared code; exercise it for every library.
    narwhals_backend_extras+=("polars" "ibis" "pyspark")
else
    if [[ $polars -eq 1 ]]; then narwhals_backend_extras+=("polars"); fi
    if [[ $ibis -eq 1 ]]; then narwhals_backend_extras+=("ibis"); fi
    if [[ $pyspark -eq 1 ]]; then narwhals_backend_extras+=("pyspark"); fi
fi

# Human-readable summary for CI logs.
{
    echo "Changed files: ${#files[@]}"
    echo "common=$common lint=$lint docs=$docs base=$base pandas=$pandas"
    echo "supplemental=$supplemental dataframe=$dataframe"
    echo "narwhals=$narwhals narwhals_backend=$narwhals_backend"
} >&2

# Machine-readable flags.
printf 'lint=%s\n' "$(boolify "$lint")"
printf 'docs=%s\n' "$(boolify "$docs")"
printf 'import_test=%s\n' "$(boolify "$import_test")"
printf 'base=%s\n' "$(boolify "$base")"
printf 'pandas=%s\n' "$(boolify "$pandas")"
printf 'supplemental=%s\n' "$(boolify "$supplemental")"
printf 'supplemental_extras=%s\n' "$(json_array ${supplemental_extras[@]+"${supplemental_extras[@]}"})"
printf 'dataframe=%s\n' "$(boolify "$dataframe")"
printf 'dataframe_extras=%s\n' "$(json_array ${dataframe_extras[@]+"${dataframe_extras[@]}"})"
printf 'narwhals_backend=%s\n' "$(boolify "$narwhals_backend")"
printf 'narwhals_backend_extras=%s\n' "$(json_array ${narwhals_backend_extras[@]+"${narwhals_backend_extras[@]}"})"
printf 'narwhals=%s\n' "$(boolify "$narwhals")"
