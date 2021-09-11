Pax Basics
==========

.. currentmodule:: pax


Random Number Generator
-----------------------

.. autosummary::

    seed_rng_key
    next_rng_key
    dropout


seed_rng_key
~~~~~~~~~~~~

.. autofunction:: seed_rng_key

    
next_rng_key
~~~~~~~~~~~~

.. autofunction:: next_rng_key


dropout
~~~~~~~

.. autofunction:: dropout



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
    Linear
    Conv1D
    Conv2D
    Conv1DTranspose
    Conv2DTranspose
    BatchNorm1D
    BatchNorm2D
    LayerNorm
    Sequential
    LSTM
    GRU
    MultiHeadAttention
    avg_pool
    max_pool




Linear
------


.. autoclass:: Linear
   :members:


Dropout
-------

.. autoclass:: Dropout
   :members:


Embed
-----

.. autoclass:: Embed
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


Recurrent
---------


LSTM
~~~~

.. autoclass:: LSTM
   :members:


GRU
~~~

.. autoclass:: GRU
   :members:


Pool
----
    
avg_pool
~~~~~~~~

.. autofunction:: avg_pool

    
max_pool
~~~~~~~~

.. autofunction:: max_pool




MultiHeadAttention
------------------

.. autoclass:: MultiHeadAttention
   :members:



Sequential
----------


.. autoclass:: Sequential
   :members:






Initializers
============

.. currentmodule:: pax.initializers

.. autosummary::

   zeros
   ones
   truncated_normal
   random_normal
   random_uniform
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


random_uniform
--------------

.. autofunction:: random_uniform


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
    scan
    build_update_fn
    dropout
    Lambda
    RngSeq


scan
----

.. autofunction:: scan


build_update_fn
---------------

.. autofunction:: build_update_fn


dropout
-------

.. autofunction:: dropout


Lambda
------

.. autoclass:: Lambda

RngSeq
------

.. autoclass:: RngSeq
   :members:

.. currentmodule:: pax.haiku

from_haiku
----------

.. autofunction:: from_haiku


See https://dm-haiku.readthedocs.io/en/latest/api.html#common-modules for more information about converted modules.    


.. currentmodule:: pax.optim


from_optax
----------

.. autofunction:: from_optax
