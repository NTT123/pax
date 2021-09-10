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

Conv1D
~~~~~~

.. autoclass:: Conv1D
   :members:

Conv2D
~~~~~~

.. autoclass:: Conv2D
   :members:

Conv1DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv1DTranspose
   :members:

Conv2DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv2DTranspose
   :members:


BatchNorm
---------

BatchNorm1D
~~~~~~~~~~~

.. autoclass:: BatchNorm1D
   :members:

BatchNorm2D
~~~~~~~~~~~

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


Initializers
============

.. currentmodule:: pax.initializers

.. autosummary::

   zeros
   ones
   truncated_normal
   random_normal
   variance_scaling
   from_haiku_initializer    



zeros
-----

.. autofunction:: zeros


ones
----

.. autofunction:: ones


truncated_normal
----------------

.. autofunction:: truncated_normal


random_normal
-------------

.. autofunction:: random_normal


variance_scaling
----------------

.. autofunction:: variance_scaling


from_haiku_initializer
----------------------

.. autofunction:: from_haiku_initializer


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
