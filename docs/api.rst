PAX Basics
==========

PAX's Module
------------

.. currentmodule:: pax

.. autoclass:: Module
   :members:

.. autoclass:: PaxFieldKind
   :members:
   
   .. autoattribute:: STATE
   .. autoattribute:: PARAMETER
   .. autoattribute:: MODULE
   .. autoattribute:: OTHERS


Purify a function
-----------------

.. currentmodule:: pax

.. autofunction:: pure


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
    GroupNorm
    Sequential
    LSTM
    GRU
    MultiHeadAttention
    Identity
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


Normalization
-------------


BatchNorm1D
~~~~~~~~~~~

.. autoclass:: BatchNorm1D
   :members:

BatchNorm2D
~~~~~~~~~~~

.. autoclass:: BatchNorm2D
   :members:



LayerNorm
~~~~~~~~~

    
.. autoclass:: LayerNorm
   :members:


GroupNorm
~~~~~~~~~

    
.. autoclass:: GroupNorm
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


Utilities
---------

Sequential
~~~~~~~~~~

.. autoclass:: Sequential
   :members:


RngSeq
~~~~~~

.. autoclass:: RngSeq
   :members:


Lambda
~~~~~~

.. autoclass:: Lambda


Identity
~~~~~~~~

.. autoclass:: Identity
   :members:

EMA
~~~

.. autoclass:: EMA
   :members:


Module Transformations
======================

.. currentmodule:: pax

A module transformation is a pure function that inputs PAX's modules and outputs PAX's modules.

.. autosummary::

   update_parameters
   update_states
   enable_train_mode
   enable_eval_mode
   select_parameters
   select_states
   freeze_parameters
   unfreeze_parameters
   transform_gradients
   apply_updates
   apply_gradients
   flatten_module
   apply_mp_policy


update_parameters
-----------------

.. autofunction:: update_parameters


update_states
-------------

.. autofunction:: update_states


enable_train_mode
-----------------

.. autofunction:: enable_train_mode


enable_eval_mode
----------------

.. autofunction:: enable_eval_mode


select_parameters
-----------------

.. autofunction:: select_parameters


select_states
-------------

.. autofunction:: select_states


freeze_parameters
-----------------

.. autofunction:: freeze_parameters


unfreeze_parameters
-------------------

.. autofunction:: unfreeze_parameters


transform_gradients
-------------------

.. autofunction:: transform_gradients


apply_updates
-------------

.. autofunction:: apply_updates


apply_gradients
---------------

.. autofunction:: apply_gradients



flatten_module
--------------

.. autoclass:: flatten_module


apply_mp_policy
---------------

.. autoclass:: apply_mp_policy
   :members: __init__




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



Utilities
=========

.. currentmodule:: pax


.. autosummary::
    grad_parameters
    scan
    build_update_fn
    dropout




grad_parameters
---------------

.. autofunction:: grad_parameters


scan
----

.. autofunction:: scan


build_update_fn
---------------

.. autofunction:: build_update_fn


dropout
-------

.. autofunction:: dropout

