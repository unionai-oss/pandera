from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

version = {}
with open("pandera/version.py") as fp:
    exec(fp.read(), version)

_extras_require = {
    "strategies": ["hypothesis >= 5.41.1"],
    "hypotheses": ["scipy"],
    "io": ["pyyaml >= 5.1", "black"],
}
extras_require = {
    **_extras_require,
    "all": list(set(x for l in _extras_require.values() for x in l)),
}

setup(
    name="pandera",
    version=version["__version__"],
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description="A light-weight and flexible validation package for pandas "
    "data structures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pandera-dev/pandera",
    project_urls={
        "Documentation": "https://pandera.readthedocs.io",
        "Issue Tracker": "https://github.com/pandera-dev/pandera/issues",
    },
    keywords=["pandas", "validation", "data-structures"],
    license="MIT",
    data_files=[("", ["LICENSE.txt"])],
    packages=["pandera"],
    install_requires=[
        "packaging",
        "numpy >= 1.9.0",
        "pandas >= 0.23.0",
        "typing_extensions ; python_version<'3.8'",
        "typing_inspect >= 0.6.0",
        "wrapt",
    ],
    extras_require=extras_require,
    python_requires=">=3.6",
    platforms="any",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)