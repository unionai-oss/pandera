# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import doctest
import sys

sys.path.insert(0, os.path.abspath("../../pandera"))


# -- Project information -----------------------------------------------------

project = "pandera"
copyright = "2019, Niels Bantilan, Nigel Markey"
author = "Niels Bantilan, Nigel Markey"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

doctest_global_setup = """
import sys
import pandas as pd
import numpy as np
pd.options.display.max_columns = None # For Travis on macOS
pd.options.display.max_rows = None # For Travis on macOS

SKIP = sys.version_info < (3, 6)
PY36 = sys.version_info < (3, 7)
"""

doctest_default_flags = (
    0
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.NORMALIZE_WHITESPACE
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autoclass_content = "both"
pygments_style = None

autodoc_default_options = {
    # 'special-members': '__call__',
    "undoc-members": False,
    # 'exclude-members': '__weakref__'
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_logo = "_static/pandera-banner-white.png"
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
    "analytics_id": "UA-71018060-2",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

rst_prolog = """
.. role:: red
.. role:: green
"""

autosummary_generate = ["API_reference.rst"]
autosummary_filename_map = {
    "pandera.Check": "pandera.Check",
    "pandera.check": "pandera.check_decorator",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
}


def setup(app):
    app.add_css_file("default.css")
