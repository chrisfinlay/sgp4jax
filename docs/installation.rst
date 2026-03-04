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

Optional dependencies
---------------------

- **test**: ``pytest``, ``sgp4`` (reference library for validation)
- **docs**: ``sphinx``, ``furo``, ``sphinx-copybutton``, ``sphinx-autodoc-typehints``

Install them with:

.. code-block:: bash

   pip install sgp4jax[test]
   pip install sgp4jax[docs]
