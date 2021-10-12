:github_url: https://github.com/ntt123/pax/tree/main/docs


.. PAX documentation master file, created by
   sphinx-quickstart on Fri Sep  3 01:09:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PAX documentation
===============================

PAX is a stateful pytree library for training neural networks using JAX. It is designed to be simple
and easy to use while preserving benefits from JAX.

.. code-block:: python

   class SGD(pax.Module):
      velocity: pax.Module
      learning_rate: float
      momentum: float 
      
      def __init__(self, params, learning_rate: float = 1e-2, momentum: float = 0.9):
         super().__init__()
         self.momentum = momentum
         self.learning_rate = learning_rate
         self.register_states('velocity', jax.tree_map(lambda x: jnp.zeros_like(x), params))
         
      def step(self, grads: pax.Module, params: pax.Module):
         self.velocity = jax.tree_map(
               lambda v, g: v * self.momentum + g * self.learning_rate,
               self.velocity,
               grads
         )
         new_params = jax.tree_map(lambda p, v: p - v, params, self.velocity)
         return new_params



Installation
------------

To install the latest version::

   pip3 install git+https://github.com/ntt123/pax.git


.. toctree::
   :caption: Guides
   :maxdepth: 1

   notebooks/basics
   notebooks/training
   notebooks/understanding
   notebooks/jax_transformations
   notebooks/limitations
   notebooks/performance



.. toctree::
   :caption: API Documentation
   :maxdepth: 1

   api




PAX is licensed under the MIT License.

Indices
=======

* :ref:`genindex`
