from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name="pandera",
    version="0.1.5",
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description="A light-weight and flexible validation package for pandas "
                "data structures.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cosmicBboy/pandera",
    project_urls={
          'Documentation': 'https://pandera.readthedocs.io',
          'Issue Tracker': 'https://github.com/cosmicBboy/pandera/issues'
      },
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
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
    platforms='any',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
        ],

)
