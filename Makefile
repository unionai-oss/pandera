.PHONY: tests clean clean-pyc upload-pypi-test upload-pypi requirements docs \
	code-cov docs-clean requirements-dev.txt

clean:
	python setup.py clean

clean-pyc:
	find . -name '*.pyc' -exec rm {} \;

upload-pypi-test:
	python setup.py sdist bdist_wheel && \
		twine upload --repository-url https://test.pypi.org/legacy/ dist/* && \
		rm -rf dist

upload-pypi:
	python setup.py sdist bdist_wheel && \
		twine upload dist/* && \
		rm -rf dist

requirements:
	pip install -r requirements-dev.txt

docs-clean:
	rm -rf docs/**/generated docs/**/methods docs/_build docs/source/_contents

docs: docs-clean
	python -m sphinx -E "docs/source" "docs/_build" && make -C docs doctest

quick-docs:
	python -m sphinx -E "docs/source" "docs/_build" -W && \
		make -C docs doctest

code-cov:
	pytest --cov-report=html --cov=pandera tests/

nox:
	nox -r --envdir .nox-virtualenv

NOX_FLAGS ?= "-r"

nox-mamba:
	nox -db mamba --envdir .nox-mamba ${NOX_FLAGS}

deps-from-conda:
	python scripts/generate_pip_deps_from_conda.py

nox-ci-requirements: deps-from-conda
	nox -db mamba --envdir .nox-mamba -s ci_requirements ${NOX_FLAGS}

nox-dev-requirements: deps-from-conda
	nox -db mamba --envdir .nox-mamba -s dev_requirements ${NOX_FLAGS}

nox-requirements: nox-ci-requirements nox-dev-requirements

nox-tests:
	nox -db mamba --envdir .nox-mamba -s tests ${NOX_FLAGS}
