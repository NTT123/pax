Pax Basics
==========

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

.. currentmodule:: pax.transforms

A module transformation is a pure function that inputs Pax's modules and outputs Pax's modules.

.. autosummary::

   mutable
   update_parameters
   update_states
   enable_train_mode
   enable_eval_mode
   select_kind
   select_parameters
   select_states
   freeze_parameters
   unfreeze_parameters
   apply_gradients
   apply_mp_policy
   apply_updates
   transform_gradients
   scan_bugs
   flatten_module


mutable
-------

.. autofunction:: mutable


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


select_kind
-----------

.. autofunction:: select_kind

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


apply_gradients
---------------

.. autofunction:: apply_gradients


apply_mp_policy
---------------

.. autoclass:: apply_mp_policy
   :members: __init__


apply_updates
-------------

.. autofunction:: apply_updates


transform_gradients
-------------------

.. autofunction:: transform_gradients



scan_bugs
---------

.. autofunction:: scan_bugs


flatten_module
--------------

.. autoclass:: flatten_module




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


Context Managers
================

.. currentmodule:: pax.ctx

.. autosummary::
    mutable
    immutable

mutable
-------

.. autoclass:: mutable



immutable
---------

.. autoclass:: immutable


Function Transformations
========================

Pax's function transformations are thin wrappers of Jax transformations. 

These wrappers enable immutable mode and additional checking to prevent potential bugs.


.. currentmodule:: pax


pax.jit
-------

``pax.jit`` is a wrapper of ``jax.jit``. 


pax.grad
--------


``pax.grad`` is a wrapper of ``jax.grad``. 



pax.vmap
--------


``pax.vmap`` is a wrapper of ``jax.vmap``. 


pax.pmap
--------


``pax.pmap`` is a wrapper of ``jax.pmap``. 



Utilities
=========

.. currentmodule:: pax.utils


.. autosummary::
    scan
    build_update_fn
    dropout


scan
----

.. autofunction:: scan


build_update_fn
---------------

.. autofunction:: build_update_fn


dropout
-------

.. autofunction:: dropout

