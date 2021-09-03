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
   :undoc-members:





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


.. currentmodule:: pax.haiku

.. autosummary::
    from_haiku
    batch_norm_2d
    layer_norm
    linear
    lstm
    gru
    embed
    conv_1d
    conv_2d
    conv_1d_transpose
    conv_2d_transpose
    avg_pool
    max_pool


    
from_haiku
----------

.. autofunction:: from_haiku


    
batch_norm_2d
-------------

.. autofunction:: batch_norm_2d

    
layer_norm
----------

.. autofunction:: layer_norm

    
linear
------

.. autofunction:: linear

    
lstm
----

.. autofunction:: lstm


    
gru
---

.. autofunction:: gru

    
embed
-----

.. autofunction:: embed

    
conv_1d
-------

.. autofunction:: conv_1d


    
conv_2d
-------

.. autofunction:: conv_2d

    
conv_1d_transpose
-----------------

.. autofunction:: conv_1d_transpose


    
conv_2d_transpose
-----------------

.. autofunction:: conv_2d_transpose

    
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
