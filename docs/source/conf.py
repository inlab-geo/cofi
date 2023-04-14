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
from sphinx_gallery.sorting import FileNameSortKey

import cofi


# -- Project information -----------------------------------------------------
project = "CoFI"
copyright = f"{datetime.date.today().year}, InLab, CoFI development team"
version = "dev" if "dev" in cofi.__version__ else f"v{cofi.__version__}"


# -- General configuration ---------------------------------------------------
sys.path.append(os.path.abspath("./_ext"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "run_sphinx_autogen",               # our own extension
    "gen_gallery_scripts",              # our own extension
    "render_cofi_gallery",              # our own extension
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build", 
    "Thumbs.db",
    ".DS_Store", 
    "README.md",
    "cofi-examples/**",
    "cofi-gallery/**",
    "**/scripts/**README.rst",
    "**/generated/**.md5",
    "**/generated/**.py",
    "**/generated/**.ipynb",
]

source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "algol_nu"        # https://pygments.org/styles/
add_function_parentheses = False

# Configuration to include links to other project docs
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "findiff": ("https://findiff.readthedocs.io/en/latest/", None),
}

# settings for the sphinx-copybutton extension
copybutton_prompt_text = ">>> "


# -- Options for HTML output -------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
html_short_title = project
html_logo = "_static/latte_art-removebg.png"
html_favicon = "_static/inlab_logo_60px.png"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/inlab-geo/cofi",
    "repository_branch": "main",
    "path_to_docs": "docs/source/",
    "launch_buttons": {
        "notebook_interface": "classic",
        "inlab_url": "http://www.inlab.edu.au/",
    },
    "extra_footer": "",
    "home_page_in_toc": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/inlab-geo/cofi",
            "icon": "https://img.shields.io/badge/GitHub-cofi-171515?logo=github&labelColor=f8f9fa&style=flat-square&logoColor=171515",
            "type": "url",
        },
        {
            "name": "Colab",
            "url": "https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/index.ipynb",
            "icon": "https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670&labelColor=f8f9fa",
            "type": "url",
        },
        {
            "name": "Version",
            "url": "https://pypi.org/project/cofi/",
            "icon": "https://img.shields.io/pypi/v/cofi?logo=pypi&style=flat-square&color=6CA8D2&labelColor=f8f9fa&label=latest",
            "type": "url",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["style.css"]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "inlab-geo", # Username
    "github_repo": "cofi", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/source/", # Path in the checkout to the docs root
}


# -- Sphinx Gallery settings --------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["examples/scripts", "tutorials/scripts"],
    "gallery_dirs": ["examples/generated", "tutorials/generated"],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": ".",
    "ignore_pattern": "._lib.py|_preprocessing.py",
    "pypandoc": True,
    "download_all_examples": False,
    "doc_module": "cofi",
}


# -- myst-nb settings ---------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
nb_execution_mode = "cache"


# -- Cutomised variables ------------------------------------------------------
rst_epilog = """
.. _repository: https://github.com/inlab-geo/cofi
.. _Slack: https://join.slack.com/t/inlab-community/shared_invite/zt-1ejny069z-v5ZyvP2tDjBR42OAu~TkHg
"""
