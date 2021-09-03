:github_url: https://github.com/ntt123/pax/tree/main/docs


.. Pax documentation master file, created by
   sphinx-quickstart on Fri Sep  3 01:09:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pax documentation
===============================

Pax is a stateful pytree library for training neural networks using Jax. It is designed to be simple
and easy to use while preserving benefits from Jax.

.. code-block:: python

   class SGD(pax.Optimizer):
      velocity: pax.Module
      learning_rate: float
      momentum: float 
      
      def __init__(self, params, learning_rate: float = 1e-2, momentum: float = 0.9):
         super().__init__()
         self.momentum = momentum
         self.learning_rate = learning_rate
         self.register_state_subtree('velocity', jax.tree_map(lambda x: jnp.zeros_like(x), params))
         
      def step(self, grads: pax.Module, model: pax.Module):
         self.velocity = jax.tree_map(
               lambda v, g: v * self.momentum + g * self.learning_rate,
               self.velocity,
               grads
         )
         params = model.parameters()
         new_params = jax.tree_map(lambda p, v: p - v, params, self.velocity)
         return model.update(new_params)



Installation
------------

To install::

   $ pip3 install git+https://github.com/ntt123/pax.git


.. toctree::
   :caption: Guides
   :maxdepth: 1

   notebooks/getting_started



.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api




Pax is licensed under the MIT License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`