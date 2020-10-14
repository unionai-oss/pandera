.PHONY: tests clean clean-pyc upload-pypi-test upload-pypi requirements docs \
	code-cov

tests:
	pytest

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

docs:
	rm -rf docs/source/generated && \
		python -m sphinx -E "docs/source" "docs/_build" && \
		make -C docs doctest

code-cov:
	pytest --cov-report=html --cov=pandera tests/
