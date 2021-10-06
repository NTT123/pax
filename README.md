<div align="left">
<img src="./images/pax_logo.png" alt="logo" width="94px"></img>
</div>

[**Introduction**](#introduction)
| [**Getting started**](#gettingstarted)
| [**Pax and others**](#paxandfriends)
| [**Examples**](https://github.com/ntt123/pax/tree/main/examples/)
| [**Modules**](#modules)
| [**Optimizers**](#optimizers)
| [**Transformations**](#transformations)
| [**Fine-tuning**](#finetune)

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

def loss_fn(model: Counter, x: jnp.ndarray):
    y = model(x)
    loss = jnp.mean(jnp.square(x - y))
    return loss, (loss, model)

grad_fn = jax.grad(loss_fn, has_aux=True, allow_int=True)

net = Counter(3)
x = jnp.array(10.)
grads, (loss, net) = grad_fn(net, x)
print(grads.counter) # (b'',)
print(grads.bias) # 60.0
```

There are a few important things in the above example:

* ``bias`` is registered as a trainable parameter using ``register_parameter`` method.
* ``counter`` is registered as a non-trainable state using ``register_state`` method.
* ``loss_fn`` returns the updated `model` in its output.
* ``allow_int=True`` to compute gradients with respect to ``model`` which contains integer ``ndarray`` leaves.

## Pax and other libraries <a id="paxandfriends"></a>

Pax module has several methods that are similar to Pytorch. 

- ``self.register_parameter(name, value)`` registers ``name`` as a trainable parameter.
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
| ``denoising_diffusion/`` | train a denoising diffusion model on `Celeb-A` dataset. |



## Modules<a id="modules"></a>

At the moment, Pax includes: 

* ``pax.nn.Embed``,
* ``pax.nn.Linear``, 
* ``pax.nn.{GRU, LSTM}``,
* ``pax.nn.{BatchNorm1D, BatchNorm2D, LayerNorm, GroupNorm}``, 
* ``pax.nn.{Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose}``, 
* ``pax.nn.{Dropout, Sequential, Identity, Lambda, RngSeq, EMA}``.

We intent to add new modules in the near future.

## Optimizers<a id="optimizers"></a>

Pax has its optimizers implemented in a separate library [opax](https://github.com/ntt123/opax). The `opax` library supports many common optimizers such as `adam`, `adamw`, `sgd`, `rmsprop`. Visit opax's github repository for more information. 


## Module transformations<a id="transformations"></a>

A module transformation is a pure function that inputs Pax's modules and outputs Pax's modules.
A Pax program can be seen as a series of module transformations.

Pax provides several module transformations:

- `pax.select_{parameters,states}`: select parameter/state leaves.
- `pax.apply_gradients`: update model & optimizer using gradients.
- `pax.update_{parameters,states}`: updates module's parameters/states.
- `pax.enable_{train,eval}_mode`: turn on/off training mode.
- `pax.(un)freeze_parameters`: freeze/unfreeze trainable parameters.
- `pax.apply_mp_policy`: apply a mixed-precision policy.


## Fine-tunning models<a id="finetune"></a>

Pax's Module provides the ``pax.freeze_parameters`` transformation to convert all trainable parameters to non-trainable states.

```python
net = pax.nn.Sequential(
    pax.nn.Linear(28*28, 64),
    jax.nn.relu,
    pax.nn.Linear(64, 10),
)

net = pax.freeze_parameters(net) 
net[-1] = pax.nn.Linear(64, 2)
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
