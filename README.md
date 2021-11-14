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

``PAX`` is a stateful [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) library for training neural networks. The main class of `PAX` is `pax.Module`.

A `pax.Module` object has two sides:

* It is a _normal_ python object which can be modified and called.
* It is a _pytree_ object whose leaves are `ndarray`s.

``pax.Module`` object manages the pytree and executes functions that depend on the pytree. As a pytree object, it can be input and output to JAX functions running on CPU/GPU/TPU cores.


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

```python
import jax, pax, jax.numpy as jnp

class Linear(pax.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    counter: jnp.ndarray

    def __init__(self):
        super().__init__()

        with self.add_parameters():
            self.weight = jnp.array(0.0)
            self.bias = jnp.array(0.0)

        with self.add_states():
            self.counter = jnp.array(0)

    def __call__(self, x):
        self.counter = self.counter + 1
        return self.weight * x + self.bias

def loss_fn(model: Linear, x: jnp.ndarray, y: jnp.ndarray):
    model, y_hat = pax.module_and_value(model)(x)
    loss = jnp.mean(jnp.square(y_hat - y))
    return loss, (loss, model)

grad_fn = jax.grad(loss_fn, has_aux=True, allow_int=True)

net = Linear()
x, y = jnp.array(1.0), jnp.array(1.0)
grads, (loss, net) = grad_fn(net, x, y)
print(grads.counter)  # (b'',)
print(grads.bias)  # -2.0
```

There are a few noteworthy points in the above example:

* ``self.weight`` and ``self.bias`` are registered as trainable parameters inside the ``add_parameters`` context.
* ``self.counter`` is registered as a non-trainable state inside the ``add_states`` context.
* ``pax.module_and_value`` transforms `model.__call__` into a 
  pure function that returns the updated model in its output.
* ``loss_fn`` returns the updated `model` in the output.
* ``jax.grad(..., allow_int=True)`` allows gradients with respect to integer ndarray leaves (e.g., `counter`).

## PAX functional programming<a id="functional"></a>

### `pax.pure`

It is a good practice to keep functions of PAX modules pure (no side effects).

Following this practice, the modifications of PAX module's internal states are restricted.
Only PAX functions decorated by `pax.pure` are allowed to modify a *copy* of its input modules.
Any modification on the copy will not affect the original inputs.
As a consequence, the only way to *update* an input module is to return it in the output.

```python
net = Linear()
net(0)
# ...
# ValueError: Cannot modify a module in immutable mode.
# Please do this computation inside a function decorated by `pax.pure`.

@pax.pure
def update_counter_wo_return(m: Linear):
    m(0)

print(net.counter)
# 0
update_counter_wo_return(net)
print(net.counter) # the same counter
# 0

@pax.pure
def update_counter_and_return(m: Linear):
    m(0)
    return m

print(net.counter)
# 0
net = update_counter_and_return(net)
print(net.counter) # increased by 1
# 1
```

### `pax.module_and_value`

It is a good practice to keep functions decorated by `pax.pure` as small as possible.

PAX provides the `pax.module_and_value` function that transforms a module's method into a pure function. The pure function also returns the updated module in its output. For example:

```python
net = Linear()
print(net.counter) # 0
net, y = pax.module_and_value(net)(0)
print(net.counter) # 1
```

In this example, `pax.module_and_value` transforms `net.__call__` into a pure function which returns the updated `net` in its output.


### Replacing parts

For convenience, PAX provides utility methods to modify a module in a functional way.

The `replace` method creates a new module with attributes replaced. 
For example, to replace `weight` and `bias` of a `pax.nn.Linear` module:

```python
fc = pax.nn.Linear(2, 2)
fc = fc.replace(weight=jnp.ones((2,2)), bias=jnp.zeros((2,)))
```

The `replace_node` method replaces a pytree node of a module:

```python
f = pax.nn.Sequential(
    pax.nn.Linear(2, 3),
    pax.nn.Linear(3, 4),
)

f = f.replace_node(f[-1], pax.nn.Linear(3, 5))
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

* ``pax.nn.Embed``,
* ``pax.nn.Linear``, 
* ``pax.nn.{GRU, LSTM}``,
* ``pax.nn.{BatchNorm1D, BatchNorm2D, LayerNorm, GroupNorm}``, 
* ``pax.nn.{Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose}``, 
* ``pax.nn.{Dropout, Sequential, Identity, Lambda, RngSeq, EMA}``.

We are intent to add new modules in the near future.

## Optimizers<a id="optimizers"></a>

PAX has its optimizers implemented in a separate library [opax](https://github.com/ntt123/opax). The `opax` library supports many common optimizers such as `adam`, `adamw`, `sgd`, `rmsprop`. Visit opax's GitHub repository for more information. 


## Module transformations<a id="transformations"></a>

A module transformation is a pure function that transforms PAX modules into new PAX modules.
A PAX program can be seen as a series of module transformations.

PAX provides several module transformations:

- `pax.select_{parameters,states}`: select parameter/state leaves.
- `pax.update_{parameters,states}`: updates module's parameters/states.
- `pax.enable_{train,eval}_mode`: turn on/off training mode.
- `pax.(un)freeze_parameters`: freeze/unfreeze trainable parameters.
- `pax.apply_mp_policy`: apply a mixed-precision policy.


## Fine-tunning models<a id="finetune"></a>

PAX's Module provides the ``pax.freeze_parameters`` transformation to convert all trainable parameters to non-trainable states.

```python
net = pax.nn.Sequential(
    pax.nn.Linear(28*28, 64),
    jax.nn.relu,
    pax.nn.Linear(64, 10),
)

net = pax.freeze_parameters(net) 
net = net.set(-1, pax.nn.Linear(64, 2))
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
