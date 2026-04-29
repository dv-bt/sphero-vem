"""Sphinx configuration for sphero-vem documentation."""

from importlib.metadata import version as _version
import inspect

# -- Project information -------------------------------------------------------
project = "sphero-vem"
author = "Davide Bottone"
copyright = "2026, Davide Bottone"

release = _version("sphero-vem")
version = release

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings ---------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings ----------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "show_version_warning_banner": True,
    "github_url": "https://github.com/dv-bt/sphero-vem",
    "navbar_end": ["navbar-icon-links"],
}


def _skip_class_attributes(app, what, name, obj, skip, options):
    # For class members, keep only callables and properties; skip data attributes.
    # `what` is the container type ("class"), not the member type, so we cannot
    # check for "attribute" here — that value never fires for class members.
    if what == "class" and not inspect.isroutine(obj) and not isinstance(obj, property):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_class_attributes)
