# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'MoonMag'
copyright = '2023, Marshall J. Styczinski'
author = 'Marshall J. Styczinski'
release = 'v1.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinxcontrib.apidoc',
              'myst_parser']
source_suffix = ['.rst', '.md']
templates_path = ['templates']

_HERE = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_HERE, '..'))
_PACKAGE_DIR = os.path.abspath(os.path.join(_HERE, '../MoonMag'))

sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _PACKAGE_DIR)

apidoc_module_dir = _ROOT_DIR
apidoc_output_dir = 'stubs'
apidoc_template_dir = 'templates'
apidoc_excluded_paths = ['configP*', 'setup.py']
apidoc_separate_modules = True
apidoc_module_first = True

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'configP*']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Install with pip install sphinx-rtd-theme
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_logo = '../misc/MoonMag_logoDocs.png'
html_favicon = '../misc/MoonMag_icon.ico'

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',  # Default is #2980B9
    'logo_only': True,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1
}
