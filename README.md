# Pax

``Pax`` is a stateful pytree library for training neural networks.

## Install

```bash
pip3 install git+https://github.com/NTT123/pax.git#egg=pax

## or test mode to run tests and examples
pip install git+https://github.com/NTT123/pax.git#egg=pax[test]
```

## Getting started

```python
import jax
import jax.numpy as jnp
import pax

class Counter(pax.Module):
    counter: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, start_value: int = 0):
        self.counter = jnp.array(start_value)
        self.register_parameter('bias', jnp.array(0.0))


    def __call__(self, x):
        self.counter = self.counter + 1
        return self.counter * x + self.bias

def loss_fn(params: Counter, model: Counter, x:  jnp.ndarray):
    model = model.update(params)
    y = model(x)
    loss = jnp.mean(jnp.square(x - y))
    return loss, (loss, model)

grad_fn = jax.grad(loss_fn, has_aux=True)

net = Counter(3)
x = jnp.array(10.)
grads, (loss, net) = grad_fn(net.parameters(), net, x)
print(grads.counter) # Nothing
print(grads.bias) # 60.0
```

There are a few important things in the above example:
1. ``counter`` is annotated with type `pax.State` to make it a non-trainable leaf of the pytree.
3. ``model = model.update(params)`` has two purposes: (1) it causes ``model`` to use ``params`` in the forward computation, (2) it returns a new version of ``model``, therefore, makes ``loss_fn`` a function without side effects.
4. ``loss_fn`` returns the updated `model` in its output.
5. ``net.parameters()`` keeps all trainable leaves intact while setting all other leaves to ``pax.tree.Nothing``. This is needed to make sure that we only compute gradients w.r.t trainable parameters only.


## Rules and limitations

Bellow are some rules and limitations that we need to be aware of when working with ``Pax``.

5. Functions (e.g., loss functions) that are input to jax's higher-order functions (e.g., ``jax.jit``, ``jax.grad``, ``jax.pmap``, etc.) should have no side effects. Modified objects/values should be returned as output of the function.

## Examples

A good way to learn about ``Pax`` is to see examples in the ``examples/`` directory:

1. ``char_rnn.py``: train a RNN Language model on TPU.
2. ``mnist.py``: train an image classifier on MNIST dataset.


## Optimizers

Pax implements optimizers in a stateful fashion. Bellow is a simple sgd optimizer with momentum.

```python
class SGD(pax.Optimizer):
    velocity: tx.Module
    momentum: float 
    learning_rate: float
    
    def __init__(self, params, learning_rate: float = 1e-2, momentum: float = 0.9):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.register_param_subtree('velocity', jax.tree_map(lambda x: jnp.zeros_like(x), params))
        
    def step(self, grads: pax.Module, model: pax.Module):
        self.velocity = jax.tree_map(
            lambda v, g: v * self.momentum + g * self.learning_rate,
            self.velocity,
            grads
        )
        params = model.parameters()
        new_params = jax.tree_map(lambda p, v: p - v, params, self.velocity)
        return model.update(new_params)
```

Because Pax's Module is stateful, ``SGD`` can store its internal state ``velocity`` naturally.

Moreover, Pax provides the ``pax.optim.from_optax`` function that convert any [optax](https://optax.readthedocs.io/en/latest/) optimizer to a pax's Module.

```python
import optax
SGD = pax.optim.from_optax(
    optax.sgd(learning_rate=learning_rate, momentum=momentum)
)
```

## Fine-tunning models.

Pax's Module provides the ``Module.filter_modules`` method to select trainable sub-modules.

```python
net = pax.nn.Sequential(
    pax.nn.Linear(28*28, 64),
    jax.nn.relu,
    pax.nn.Linear(64, 10),
)

def finetune_filter(mod, info):
    model = info['parent']
    old_mod = info['old']
    if isinstance(model, pax.nn.Sequential):
        trainable_layers = [model.modules[2]]
        if old_mod in trainable_layers:
            return mod
        else:
            return jax.tree_map(lambda x: tree.Nothing(), mod)
    else:
        return mod

# we only predict two classes.
net.modules[2] = pax.nn.Linear(64, 2)
net = net.filter_modules(finetune_filter)
```
``net.parameters()`` will only returns parameters of the last layer that we want to finetune.
