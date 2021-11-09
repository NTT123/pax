PAX Basics
==========

.. currentmodule:: pax

.. autosummary::
    Module
    pure
    module_and_value
    seed_rng_key
    next_rng_key



PAX's Module
------------

.. currentmodule:: pax

.. autoclass:: Module
   :inherited-members:
   :members:


.. autoclass:: ParameterModule
   :members:


.. autoclass:: StateModule
   :members:


.. autoclass:: PaxKind
   :members:
   
   .. autoattribute:: STATE
   .. autoattribute:: PARAMETER
   .. autoattribute:: MODULE
   .. autoattribute:: UNKNOWN



Purify functions and methods
----------------------------

.. currentmodule:: pax

.. autofunction:: pure

.. autofunction:: module_and_value 


Random Number Generator
-----------------------

.. autosummary::

    seed_rng_key
    next_rng_key


seed_rng_key
~~~~~~~~~~~~

.. autofunction:: seed_rng_key

    
next_rng_key
~~~~~~~~~~~~

.. autofunction:: next_rng_key
