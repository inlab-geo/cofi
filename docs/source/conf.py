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

import cofi

# Check if building on Read the Docs
on_rtd = os.environ.get("READTHEDOCS") == "True"


# ---------------------------------------------------------------------------
# Custom Sphinx Gallery sort key: respects dependencies.txt ordering
# ---------------------------------------------------------------------------
class DependencySortKey:
    """Sort gallery scripts by the order declared in _ordering.txt.

    gen_gallery_scripts.py writes an ``_ordering.txt`` file into each
    gallery script directory.  Scripts listed there are executed first
    (in the listed order); any remaining scripts fall back to alphabetical
    order after them.
    """

    def __init__(self, src_dir):
        self.order = {}
        order_file = os.path.join(src_dir, "_ordering.txt")
        if os.path.isfile(order_file):
            with open(order_file) as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.order[name] = i

    def __call__(self, filename):
        # Scripts in _ordering.txt get their declared index;
        # unknown scripts sort after them, alphabetically.
        return (self.order.get(filename, len(self.order)), filename)


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
    "render_cofi_gallery",              # our own extension
]

# On RTD, skip sphinx-gallery execution - use pre-committed outputs
if not on_rtd:
    extensions.append("gen_gallery_scripts")        # converts notebooks to gallery scripts
    extensions.append("sphinx_gallery.gen_gallery") # executes gallery scripts

templates_path = ["_templates"]

exclude_patterns = [
    "_build", 
    "Thumbs.db",
    ".DS_Store", 
    "README.md",
    "cofi-examples/**",
    "cofi-gallery/**",
    "examples/README.rst", 
    "examples/scripts_field_data/README.rst", 
    "examples/scripts_synth_data/README.rst", 
    "tutorials/scripts/README.rst", 
    "**/generated/**.md5",
    "**/generated/**.py",
    "**/generated/**.ipynb",
    "data/**",
    "theory/**",
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
html_title = f"{project} <span class='project-version'>{version}</span>"
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

# -- Patch faulthandler for Sphinx Gallery compatibility with pyfm2d --------
# pyfm2d calls faulthandler.enable() at import time, which requires a real
# file descriptor on stderr. Sphinx Gallery replaces stderr with a StringIO
# that lacks fileno(), causing an UnsupportedOperation error.
import faulthandler
_original_fh_enable = faulthandler.enable
def _safe_faulthandler_enable(*args, **kwargs):
    try:
        return _original_fh_enable(*args, **kwargs)
    except Exception:
        pass
faulthandler.enable = _safe_faulthandler_enable

# -- Sphinx Gallery settings --------------------------------------------------
# Sphinx Gallery settings (only when not on RTD)
if not on_rtd:
    sphinx_gallery_conf = {
        "examples_dirs": ["examples", "tutorials/scripts"],
        "gallery_dirs": ["examples/generated", "tutorials/generated"],
        "within_subsection_order": DependencySortKey,
        "filename_pattern": ".",
        "ignore_pattern": "._lib.py|_preprocessing.py|xrayTomography.py|neptune_deterministic_methods.py|setup_inversion.py|sw_tomography.py|fmm_tomography.py|neptune_bayesian_methods.py",
        "pypandoc": True,
        "download_all_examples": False,
        "doc_module": "cofi",
        "run_stale_examples": False,
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
