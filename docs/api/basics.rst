PAX Basics
==========

.. currentmodule:: pax

.. autosummary::
    Module
    EmptyNode
    pure
    purecall
    seed_rng_key
    next_rng_key



PAX's Module
------------

.. currentmodule:: pax

.. autoclass:: Module
   :members:
      __init__,
      parameters,
      training,
      train,
      eval,
      update_parameters,
      replace,
      replace_node,
      summary,
      apply,
      state_dict,
      load_state_dict,
      __or__,
      __mod__



.. autoclass:: ParameterModule
   :members:


.. autoclass:: StateModule
   :members:


.. autoclass:: EmptyNode
   :members:


Purify functions and methods
----------------------------

.. currentmodule:: pax

.. autofunction:: pure

.. autofunction:: purecall


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
