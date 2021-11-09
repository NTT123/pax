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
   apply_mp_policy
   unwrap_mp_policy

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


apply_mp_policy
---------------

.. autofunction:: apply_mp_policy


unwrap_mp_policy
----------------

.. autofunction:: unwrap_mp_policy
