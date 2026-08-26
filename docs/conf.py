"""Sphinx configuration for sgp4jax."""

from importlib.metadata import version as _version

# sgp4jax requires JAX double precision and raises on import without it, so
# enable it before autodoc imports the package.
import jax

jax.config.update("jax_enable_x64", True)

project = "sgp4jax"
copyright = "2024, sgp4jax contributors"
author = "sgp4jax contributors"
release = _version("sgp4jax")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", ".ipynb_checkpoints"]

html_theme = "furo"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
