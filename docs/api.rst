Pax Basics
==========

.. currentmodule:: pax


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



Pax's Module
------------
.. currentmodule:: pax.module


.. autoclass:: Module
   :members:


.. autoclass:: PaxFieldKind
   :members:
   
   .. autoattribute:: STATE
   .. autoattribute:: PARAMETER
   .. autoattribute:: MODULE
   .. autoattribute:: STATE_SUBTREE
   .. autoattribute:: PARAMETER_SUBTREE
   .. autoattribute:: MODULE_SUBTREE
   .. autoattribute:: OTHERS




Common Modules
==============

.. currentmodule:: pax.nn

.. autosummary::
    MultiHeadAttention
    BatchNorm
    BatchNorm1D
    BatchNorm2D
    Conv1D
    Conv2D
    Conv1DTranspose
    Conv2DTranspose
    LayerNorm
    Linear
    Sequential




Linear
------


.. autoclass:: Linear
   :members:



Convolution
-----------

.. autoclass:: Conv1D
   :members:

.. autoclass:: Conv2D
   :members:

.. autoclass:: Conv1DTranspose
   :members:

.. autoclass:: Conv2DTranspose
   :members:


BatchNorm
---------

.. autoclass:: BatchNorm
   :members:

.. autoclass:: BatchNorm1D
   :members:

.. autoclass:: BatchNorm2D
   :members:



LayerNorm
---------

    
.. autoclass:: LayerNorm
   :members:



MultiHeadAttention
------------------

.. autoclass:: MultiHeadAttention
   :members:



Sequential
----------


.. autoclass:: Sequential
   :members:



Haiku Modules
=============


.. currentmodule:: pax.haiku

.. autosummary::
    from_haiku
    lstm
    gru
    embed
    avg_pool
    max_pool

    
from_haiku
----------

.. autofunction:: from_haiku


See https://dm-haiku.readthedocs.io/en/latest/api.html#common-modules for more information about converted modules.


    
lstm
----

.. autofunction:: lstm


gru
---

.. autofunction:: gru

    
embed
-----

.. autofunction:: embed

    
avg_pool
--------

.. autofunction:: avg_pool

    
max_pool
--------

.. autofunction:: max_pool

    

Optimizer
=========

.. currentmodule:: pax.optim


.. autosummary::

    from_optax


from_optax
----------

.. autofunction:: from_optax




Utilities
=========

.. currentmodule:: pax.utils


.. autosummary::

    build_update_fn
    Lambda
    RngSeq


build_update_fn
---------------

.. autofunction:: build_update_fn

Lambda
------

.. autoclass:: Lambda

RngSeq
------

.. autoclass:: RngSeq
   :members:
