<div align="left">
<img src="https://raw.githubusercontent.com/NTT123/pax/main/images/pax_logo.png" alt="logo" width="94px"></img>
</div>

[**Introduction**](#introduction)
| [**Getting started**](#gettingstarted)
| [**Functional programming**](#functional)
| [**Examples**](https://github.com/ntt123/pax/tree/main/examples/)
| [**Modules**](#modules)
| [**Fine-tuning**](#finetune)

![pytest](https://github.com/ntt123/pax/workflows/pytest/badge.svg)
![docs](https://readthedocs.org/projects/pax/badge/?version=main)
![pypi](https://img.shields.io/pypi/v/pax3)


## Introduction<a id="introduction"></a>

PAX is a [JAX]-based library for training neural networks.

PAX modules are registered as JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html), therefore, they can be input or output of JAX transformations such as `jax.jit`, `jax.grad`, etc. This makes programming with modules very convenient and easy to understand.

## Installation<a id="installation"></a>

Install from PyPI:

```bash
pip install pax3
```

Or install the latest version from Github:

```bash
pip install git+https://github.com/ntt123/pax.git

## or test mode to run tests and examples
pip install git+https://github.com/ntt123/pax.git#egg=pax3[test]
```


## Getting started<a id="gettingstarted"></a>


Below is a simple example of a `Linear` module.

```python
import jax.numpy as jnp
import pax

class Linear(pax.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    parameters = pax.parameters_method("weight", "bias")

    def __init__(self):
        super().__init__()
        self.weight = jnp.array(0.0)
        self.bias = jnp.array(0.0)

    def __call__(self, x):
        return self.weight * x + self.bias
```

The implementation is very similar to a normal python class. However, we need an additional line

```python
    parameters = pax.parameters_method("weight", "bias")
```

to declare that `weight` and `bias` are *trainable parameters* of the Linear module.

## PAX functional programming<a id="functional"></a>

### `pax.pure`

A PAX module can have internal states. For example, below is a simple `Counter` module with an internal counter.

```python
class Counter(pax.Module):
    count : jnp.ndarray

    def __init__(self):
        super().__init__()
        self.count = jnp.array(0)
    
    def __call__(self):
        self.count = self.count + 1
        return self.count
```

However, PAX *aims* to guarantee that modules will have no side effects from the outside point of view.
Therefore, the modifications of these internal states are restricted. For example, we get an error when trying to call `Counter` directly.

```python
counter = Counter()
count = counter()
# ...
# ----> 9         self.count = self.count + 1
# ...
# ValueError: Cannot modify a module in immutable mode.
# Please do this computation inside a function decorated by `pax.pure`.
```

Only functions decorated by `pax.pure` are allowed to modify input module's internal states.

```python
@pax.pure
def update_counter(counter: Counter):
    count = counter()
    return counter, count

counter, count = update_counter(counter)
print(counter.count, count)
# 1 1
```

Note that we have to return `counter` in the output of `update_counter`, otherwise, the `counter` object will not be updated. This is because `pax.pure` only provides `update_counter` a copy of the `counter` object.


### `pax.purecall`

For convenience, PAX provides the `pax.purecall` function. 
It is a shortcut for `pax.pure(lambda f, x: [f, f(x)])`.

Instead of implementing an `update_counter` function, we can do the same thing with:

```python
counter, count = pax.purecall(counter)
print(counter.count, count)
# 2, 2
```

### Replacing parts

PAX provides utility methods to modify a module in a functional way.

The `replace` method creates a new module with attributes replaced. 
For example, to replace `weight` and `bias` of a `pax.Linear` module:

```python
fc = pax.Linear(2, 2)
fc = fc.replace(weight=jnp.ones((2,2)), bias=jnp.zeros((2,)))
```

The `replace_node` method replaces a pytree node of a module:

```python
f = pax.Sequential(
    pax.Linear(2, 3),
    pax.Linear(3, 4),
)

f = f.replace_node(f[-1], pax.Linear(3, 5))
print(f.summary())
# Sequential
# ├── Linear(in_dim=2, out_dim=3, with_bias=True)
# └── Linear(in_dim=3, out_dim=5, with_bias=True)
```

## PAX and other libraries <a id="paxandfriends"></a>

PAX learns a lot from other libraries:
- PAX borrows the idea that _a module is also a pytree_ from [treex] and [equinox]. 
- PAX uses the concept of _trainable parameters_ and _non-trainable states_ from [dm-haiku].
- PAX has similar methods to PyTorch such as `model.apply()`, `model.parameters()`, `model.eval()`, etc.
- PAX uses [objax]'s approach to implement optimizers as modules. 
- PAX uses [jmp] library for supporting mixed precision. 
- And of course, PAX is heavily influenced by [jax] functional programming approach.


## Examples<a id="examples"></a>

A good way to learn about ``PAX`` is to see examples in the [examples/](./examples) directory.


<details>
<summary>Click to expand</summary>

| Path     |      Description      |
|----------|-----------------------|
| ``char_rnn.py``  |  train a RNN language model on TPU.             |
| ``transformer/`` |    train a Transformer language model on TPU.   |
| ``mnist.py``     | train an image classifier on `MNIST` dataset.   |
| ``notebooks/VAE.ipynb``   | train a variational autoencoder.       |
| ``notebooks/DCGAN.ipynb`` | train a DCGAN model on `Celeb-A` dataset. |
| ``notebooks/fine_tuning_resnet18.ipynb``    | finetune a pretrained ResNet18 model on `cats vs dogs` dataset. |
| ``notebooks/mixed_precision.ipynb`` | train a U-Net image segmentation with mixed precision. |
| ``mnist_mixed_precision.py`` | train an image classifier with mixed precision. |
| ``wave_gru/`` | train a WaveGRU vocoder: convert mel-spectrogram to waveform. |
| ``denoising_diffusion/`` | train a denoising diffusion model on `Celeb-A` dataset. |

</details>




## Modules<a id="modules"></a>

At the moment, PAX includes: 

* ``pax.Embed``,
* ``pax.Linear``, 
* ``pax.{GRU, LSTM}``,
* ``pax.{BatchNorm1D, BatchNorm2D, LayerNorm, GroupNorm}``, 
* ``pax.{Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose}``, 
* ``pax.{Dropout, Sequential, Identity, Lambda, RngSeq, EMA}``.

## Optimizers<a id="optimizers"></a>

PAX has its optimizers implemented in a separate library [opax](https://github.com/ntt123/opax). The `opax` library supports many common optimizers such as `adam`, `adamw`, `sgd`, `rmsprop`. Visit opax's GitHub repository for more information. 


## Fine-tunning models<a id="finetune"></a>

PAX's Module provides the ``pax.freeze_parameters`` transformation to convert all trainable parameters to non-trainable states.

```python
net = pax.Sequential(
    pax.Linear(28*28, 64),
    jax.nn.relu,
    pax.Linear(64, 10),
)

net = pax.freeze_parameters(net) 
net = net.set(-1, pax.Linear(64, 2))
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
