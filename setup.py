from setuptools import setup


setup(
    name="pandera",
    author="Niels Bantilan",
    author_email="niels.bantilan",
    version="0.0.1",
    description="A pandas data structure validation library",
    url="https://github.com/cosmicBboy/pandera",
    install_requires=[
        "numpy >= 1.9.0",
        "pandas >= 0.23.0",
        "schema >= 0.6.8",
    ]
)
