from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name="pandera",
    version="0.1.2",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description = 'A light-weight and flexible validation package for pandas data structures.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cosmicBboy/pandera",
    keywords=["pandas", "validation", "data-structures"],
    license="MIT",
    packages=[
        "pandera",
    ],
    install_requires=[
        "enum34 ; python_version<'3.4'",
        "numpy >= 1.9.0",
        "pandas >= 0.23.0",
        "wrapt",
        "scipy ; python_version<'2.7'",
    ],
    python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    platforms='any',
)