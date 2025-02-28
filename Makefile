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

.PHONY: install-uv
install-uv:
	pip install uv

setup: install-uv
	uv sync --all-extras

setup-macos: install-uv
	uv sync --all-extras
	uv pip install polars-lts-cpu

docs-clean:
	rm -rf docs/source/reference/generated docs/**/generated docs/**/methods docs/_build docs/source/_contents

docs: docs-clean
	python -m sphinx -W -E "docs/source" "docs/_build" && make -C docs doctest

quick-docs:
	python -m sphinx -E "docs/source" "docs/_build" && make -C docs doctest

code-cov:
	pytest --cov-report=html --cov=pandera tests/

NOX_FLAGS ?= "-r"

deps-from-environment.yml:
	python scripts/generate_pip_deps_from_conda.py

unit-tests:
	pytest tests/core

nox-tests:
	nox -db uv -s tests ${NOX_FLAGS}
