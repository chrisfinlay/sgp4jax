Installation
============

From PyPI
---------

.. code-block:: bash

   pip install sgp4jax

From source
-----------

.. code-block:: bash

   git clone https://github.com/chrisfinlay/sgp4jax.git
   cd sgp4jax
   pip install -e ".[test]"

Enabling double precision
-------------------------

sgp4jax requires JAX double precision, which JAX leaves off by default.
Enable it before the first JAX operation in your program:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   import sgp4jax

or set the environment variable before starting Python:

.. code-block:: bash

   export JAX_ENABLE_X64=1

Importing sgp4jax without it raises :exc:`RuntimeError`.  See
:ref:`double-precision` for the rationale and for what this means for the
dtype of your own arrays.

Optional dependencies
---------------------

- **test**: ``pytest``, ``sgp4`` (reference library for validation)
- **docs**: ``sphinx``, ``furo``, ``sphinx-copybutton``, ``sphinx-autodoc-typehints``

Install them with:

.. code-block:: bash

   pip install sgp4jax[test]
   pip install sgp4jax[docs]
