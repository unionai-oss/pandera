#!/bin/bash
set -ev

run_ci() {
    conda create -q -y -n pandera-ci-env-$1 python=$1 && \
        source activate pandera-ci-env-$1 && \
        pip install -e . && \
        pytest && \
        source deactivate && \
        conda remove -q -y -n pandera-ci-env-$1 --all
}

run_ci 2.7
run_ci 3.5
run_ci 3.6
run_ci 3.7
