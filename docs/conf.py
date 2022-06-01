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
import os
import datetime
import sys
import subprocess

import cofi


# -- Generate API references doc ---------------------------------------------
def run_autogen(_):
    cmd_path = "sphinx-autogen"
    if hasattr(sys, "real_prefix"):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, "bin", cmd_path))
    subprocess.check_call(
        [cmd_path, "-i", "-t", "_templates", "-o", "api/generated", "api/index.rst"]
    )


def setup(app):
    app.connect("builder-inited", run_autogen)
    app.registry.source_suffix.pop(".ipynb", None)      # Ignore .ipynb files


# -- Project information -----------------------------------------------------
project = "CoFI"
copyright = f"{datetime.date.today().year}, InLab, CoFI development team"
version = "dev" if "dev" in cofi.__version__ else f"v{cofi.__version__}"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_panels",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    # "nbsphinx",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.mermaid",
    "sphinxemoji.sphinxemoji",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build", 
    "Thumbs.db",
    ".DS_Store", 
    "README.md",
    "cofi-examples/*.md",
    "cofi-examples/scripts/README.rst",
]

source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "trac"        # https://pygments.org/styles/
add_function_parentheses = False

# Configuration to include links to other project docs
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
}

# Disable including boostrap CSS for sphinx_panels since it's already included
# with sphinx-book-theme
panels_add_bootstrap_css = False
panels_css_variables = {
    "tabs-color-label-inactive": "hsla(231, 99%, 66%, 0.5)",
}

# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "


# -- Options for HTML output -------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
# html_logo = "_static/cofi-logo-removebg.png"
html_logo = "_static/latte_art-removebg.png"
html_favicon = "_static/inlab_logo_60px.png"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/inlab-geo/cofi",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "launch_buttons": {
        "notebook_interface": "classic",
        "inlab_url": "http://www.inlab.edu.au/",
    },
    "extra_footer": "",
    "home_page_in_toc": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

html_static_path = ["_static"]
html_css_files = ["style.css"]


# -- Sphinx Gallery settings --------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "cofi-examples/scripts",
    "gallery_dirs": "cofi-examples/generated",
    "filename_pattern": ".",
    "ignore_pattern": "._lib.py",
    "pypandoc": True,
    "download_all_examples": False,
}


# -- Cutomised variables ------------------------------------------------------
rst_epilog = """
.. _repository: https://github.com/inlab-geo/cofi
"""
