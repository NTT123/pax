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


.. currentmodule:: pax.haiku


Haiku
-----

.. autosummary::

    dynamic_unroll
    from_haiku


from_haiku
~~~~~~~~~~

.. autofunction:: from_haiku

    
dynamic_unroll
~~~~~~~~~~~~~~

.. autofunction:: dynamic_unroll




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
    LayerNorm
    Linear
    Sequential




Linear
------

    
.. autoclass:: Linear


Sequential
----------


.. autoclass:: Sequential



LayerNorm
---------

    
.. autoclass:: LayerNorm



MultiHeadAttention
------------------

.. autoclass:: MultiHeadAttention


BatchNorm
---------

.. autoclass:: BatchNorm

.. autoclass:: BatchNorm1D

.. autoclass:: BatchNorm2D


Convolution
-----------

.. autoclass:: Conv1D

.. autoclass:: Conv2D


Haiku Modules
=============

.. automodule:: pax.haiku




Optimizer
=========

.. currentmodule:: pax.optim


.. autosummary::

    Optimizer
    from_optax
    adamw
    OptaxState
    

Optimizer
---------


.. autoclass:: Optimizer


from_optax
----------

.. autofunction:: from_optax

adamw
-----

.. autofunction:: adamw




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
