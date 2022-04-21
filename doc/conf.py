# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxarg.ext",
    "sphinx_markdown_tables",
    "sphinx.ext.autosummary",
    'sphinx.ext.doctest',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:


# enable to markdown


# source_parsers = {
#     '.md' : CommonMarkParser,
# }

# from recommonmark.transform import AutoStructify

# github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/doc/'

# def setup(app):
#     app.add_config_value('recommonmark_config',{
#         'url_resolver': lambda url: github_doc_root + url,
#         'auto_toc_tree_section': 'Contents',
#     }, True)
#     app.add_transform_value(AutoStructify)

# The master toctree document.
# master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'CATEX'
copyright = '2021, qiangqiang wang'
author = 'qiangqiang wang'


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true , `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#TODO: We could try another theme.
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']

# Custom sidebar templates, must be a dictionary that maps document names 
# to template names.
#
# This is required for the alabaster theme 
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html', # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

from recommonmark.parser import CommonMarkParser
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
source_suffix = ['.rst', '.md']

# TODO: add options for HTMLhelp , latex, etc.