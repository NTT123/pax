Experimental
============

.. currentmodule:: pax.experimental


.. autosummary::
    Flattener
    LazyModule
    graph.build_graph_module
    default_mp_policy
    apply_scaled_gradients


Flattener
---------

.. autoclass:: Flattener
   :members:


Graph API
---------

.. currentmodule:: pax.experimental.graph

.. autoclass:: Node
   :members:

.. autoclass:: InputNode
   :members:

.. autoclass:: GraphModule
   :members:

.. autofunction:: build_graph_module


Lazy Module
-----------

.. currentmodule:: pax.experimental

.. autoclass:: LazyModule
   :members:


Mixed Precision
---------------

.. currentmodule:: pax.experimental

.. autofunction:: default_mp_policy
.. autofunction:: apply_scaled_gradients
