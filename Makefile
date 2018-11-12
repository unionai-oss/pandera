.PHONY: tests upload-pypi

tests:
	pytest

clean:
	python setup.py clean

clean-pyc:
	find . -name '*.pyc' -exec rm {} \;

upload-pypi:
	python setup.py sdist upload -r pypi

requirements:
	pip install -r requirements.txt

mock-ci-tests:
	. ./ci_tests.sh
