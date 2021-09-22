<div align="left">
<img src="./images/pax_logo.png" alt="logo" width="94px"></img>
</div>

[**Introduction**](#introduction)
| [**Getting started**](#gettingstarted)
| [**Pax and others**](#paxandfriends)
| [**Examples**](https://github.com/ntt123/pax/tree/main/examples/)
| [**Modules**](#modules)
| [**Fine-tuning**](#finetune)
| [**Documentation**](https://pax.readthedocs.io/en/main)

![pytest](https://github.com/ntt123/pax/workflows/pytest/badge.svg)
![docs](https://readthedocs.org/projects/pax/badge/?version=main)
![pypi](https://img.shields.io/pypi/v/pax-j)


## Introduction<a id="introduction"></a>

``Pax`` is a stateful [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) library for training neural networks. The central object of `Pax` is a `pax.Module`.

A  `pax.Module` has two sides:

* It is a _normal_ python object which can be modified and called (it has ``__call__`` method).
* It is a _pytree_ object whose leaves are `ndarray`'s.

``pax.Module`` manages the pytree and executes functions that depends on the pytree. As a pytree object, `pax.Module` can be input and output to jax functions running on CPU/GPU/TPU cores.


## Installation<a id="installation"></a>

```bash
pip3 install git+https://github.com/ntt123/pax.git

## or test mode to run tests and examples
pip3 install git+https://github.com/ntt123/pax.git#egg=pax[test]
```


## Getting started<a id="gettingstarted"></a>

```python
import jax
import jax.numpy as jnp
import pax

class Counter(pax.Module):
    def __init__(self, start_value: int = 0):
        super().__init__()
        self.register_parameter("bias", jnp.array(0.0))
        self.register_state("counter", jnp.array(start_value))


    def __call__(self, x):
        self.counter = self.counter + 1
        return self.counter * x + self.bias

def loss_fn(params: Counter, model: Counter, x:  jnp.ndarray):
    model = model.update(params)
    y = model(x)
    loss = jnp.mean(jnp.square(x - y))
    return loss, (loss, model)

grad_fn = pax.grad(loss_fn, has_aux=True)

net = Counter(3)
x = jnp.array(10.)
grads, (loss, net) = grad_fn(net.parameters(), net, x)
print(grads.counter) # None
print(grads.bias) # 60.0
```

There are a few important things in the above example:

* ``bias`` is registered as a trainable parameter using ``register_parameter`` method.
* ``counter`` is registered as a non-trainable state using ``register_state`` method.
* ``model = model.update(params)`` causes ``model`` to use ``params`` in the forward computation.
* ``loss_fn`` returns the updated `model` in its output.
* ``pax.grad`` is a wrapper of `jax.grad` with immutable mode turned on and additional checks to prevent potential bugs.
* ``net.parameters()`` returns a copy of `net` as such keeping all trainable leaves intact while setting all other leaves to ``None``. 
This is needed to make sure that we only compute gradients w.r.t trainable parameters.


## Pax and other libraries <a id="paxandfriends"></a>

Pax module has several methods that are similar to Pytorch. 

- ``self.parameters()`` returns parameters of the module.
- ``self.register_parameter(name, value)`` registers ``name`` as a trainable parameters.
- ``self.register_module(name, mod)`` registers ``mod`` as a submodule of ``self``.
- ``self.apply(func)`` applies ``func`` on all modules of ``self`` recursively.
- ``self.train()`` and ``self.eval()`` returns a new module in ``train/eval`` mode.
- ``self.training`` returns if ``self`` is in training mode.

Pax learns a lot from other libraries too:
- Pax borrows the idea that _a module is also a pytree_ from [treex] and [equinox]. 
- Pax uses the concept of _trainable parameters_ and _non-trainable states_ from [dm-haiku].
- Pax uses [objax]'s approach to implement optimizers as modules. 
- Pax uses [jmp] library for supporting mixed precision. 
- And of course, Pax is heavily influenced by [jax] functional programming approach.


## Examples<a id="examples"></a>

A good way to learn about ``Pax`` is to see examples in the [examples/](./examples) directory:


| Path     |      Description      |
|----------|-----------------------|
| ``char_rnn.py``  |  train a RNN language model on TPU.             |
| ``transformer/`` |    train a Transformer language model on TPU.   |
| ``mnist.py``     | train an image classifier on `MNIST` dataset.   |
| ``notebooks/VAE.ipynb``   | train a variational autoencoder.       |
| ``notebooks/DCGAN.ipynb`` | train a DCGAN model on `Celeb-A` dataset. |
| ``notebooks/fine_tuning_resnet18.ipynb``    | finetune a pretrained ResNet18 model on `cats vs dogs` dataset. |
| ``notebooks/mixed_precision.ipynb`` | train a U-Net image segmentation with mixed precision. |
| ``mnist_mixed_precision.py`` (experimental) | train an image classifier with mixed precision. |
| ``wave_gru/`` | train a WaveGRU vocoder: convert mel-spectrogram to waveform. |



## Modules<a id="modules"></a>

At the moment, Pax includes: 

* ``pax.nn.Linear``, 
* ``pax.nn.{BatchNorm1D, BatchNorm2D, LayerNorm, GroupNorm}``, 
* ``pax.nn.{Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose}``, 
* ``pax.nn.{GRU, LSTM}``,
* ``pax.nn.{Sequential, Identity, Lambda, RngSeq, EMA}``

We intent to add new modules in the near future.

## Optimizers<a id="optimizers"></a>

Pax has its optimizers implemented in a separate library [opax](https://github.com/ntt123/opax). The `opax` library supports many common optimizers such as `adam`, `adamw`, `sgd`, `rmsprop`. Visit opax's github repository for more information. 


## Fine-tunning models<a id="finetune"></a>

Pax's Module provides the ``freeze`` method to convert all trainable parameters to non-trainable states.

```python
net = pax.nn.Sequential(
    pax.nn.Linear(28*28, 64),
    jax.nn.relu,
    pax.nn.Linear(64, 10),
)

# freeze all parameters.
net = net.freeze() 

# replace the last layer by a new module.
net.modules[-1] = pax.nn.Linear(64, 2)
```

After this, ``net.parameters()`` will only return trainable parameters of the last layer.


[jax]: https://github.com/google/jax
[objax]: https://github.com/google/objax
[dm-haiku]: https://github.com/deepmind/dm-haiku
[optax]: https://github.com/deepmind/optax
[jmp]: https://github.com/deepmind/jmp
[pytorch]: https://github.com/pytorch/pytorch
[treex]: https://github.com/cgarciae/treex
[equinox]: https://github.com/patrick-kidger/equinox
