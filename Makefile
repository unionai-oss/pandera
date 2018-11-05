.PHONY: tests upload-pypi

tests:
	pytest

clean:
	python setup.py clean

clean_pyc:
	find . -name '*.pyc' -exec rm {} \;

upload-pypi:
	python setup.py sdist upload -r pypi
