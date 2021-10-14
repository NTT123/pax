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
* It is a _pytree_ object whose leaves are `ndarray`'s.

``pax.Module`` object manages the pytree and executes functions that depend on the pytree. As a pytree object, it can be input and output to JAX functions running on CPU/GPU/TPU cores.


## Installation<a id="installation"></a>

Install from PyPI:

```bash
pip3 install pax3
```

Or install the latest version from Github:

```bash
pip3 install git+https://github.com/ntt123/pax.git

## or test mode to run tests and examples
pip3 install git+https://github.com/ntt123/pax.git#egg=pax3[test]
```


## Getting started<a id="gettingstarted"></a>

```python
import jax
import jax.numpy as jnp
import pax

class Counter(pax.Module):
    bias: jnp.ndarray
    counter: jnp.ndarray
    
    def __init__(self, start_value: int = 0):
        super().__init__()
        self.register_parameter("bias", jnp.array(0.0))
        self.register_state("counter", jnp.array(start_value))

    def __call__(self, x):
        self.counter = self.counter + 1
        return self.counter * x + self.bias

def loss_fn(model: Counter, x: jnp.ndarray):
    model, y = pax.module_and_value(model)(x)
    loss = jnp.mean(jnp.square(x - y))
    return loss, (loss, model)

grad_fn = jax.grad(loss_fn, has_aux=True, allow_int=True)

net = Counter(3)
x = jnp.array(10.)
grads, (loss, net) = grad_fn(net, x)
print(grads.counter) # (b'',)
print(grads.bias) # 60.0
```

There are few noteworthy points in the above example:

* ``bias`` is registered as a trainable parameter using ``register_parameter`` method.
* ``counter`` is registered as a non-trainable state using ``register_state`` method.
* ``pax.module_and_value`` transforms `model.__call__` into a 
  pure function that returns the updated model in its output.
* ``loss_fn`` returns the updated `model` in the output.
* ``allow_int=True`` to compute gradients with respect to integer ndarray leaf `counter`.

## PAX functional programming<a id="functional"></a>

### `pax.pure`

Let "PAX function" mean functions whose inputs contain PAX modules.

It is a good practice to make PAX functions pure (no side effects).

Even though PAX modules are stateful objects, the modifications of PAX module's internal states are restricted. 
Only PAX functions decorated by `pax.pure` are allowed to modify PAX modules.

```python
net = Counter(3)
net(0)
# ...
# ValueError: Cannot modify a module in immutable mode.
# Please do this computation inside a function decorated by `pax.pure`.
```

Furthermore, a decorated function can only access a copy of its inputs. Any modification on the copy will not affect the original inputs.

```python
@pax.pure
def update_counter_wo_return(m: Counter):
    m(0)

print(net.counter)
# 3
update_counter_wo_return(net)
print(net.counter) # the same counter
# 3
```

As a consequence, the only way to *update* an input module is to return it in the output.

```python
@pax.pure
def update_counter(m: Counter):
    m(0)
    return m

print(net.counter)
# 3
net = update_counter(net)
print(net.counter) # increased by 1
# 4
```

### `pax.module_and_value`

It is a good practice to keep functions decorated by `pax.pure` as small as possible.

PAX provides the function `pax.module_and_value` that transforms a module's method into a pure function. The pure function also returns the updated module in its output. For example:

```python
net = Counter(3)
print(net.counter) # 3
net, y = pax.module_and_value(net)(0)
print(net.counter) # 4
```

In this example, `pax.module_and_value` transforms `net.__call__` into a pure function which returns the updated `net` in its output.


## PAX and other libraries <a id="paxandfriends"></a>

PAX module has several methods that are similar to Pytorch. 

- ``self.register_parameter(name, value)`` registers ``name`` as a trainable parameter.
- ``self.apply(func)`` applies ``func`` on all modules of ``self`` recursively.
- ``self.train()`` and ``self.eval()`` returns a new module in ``train/eval`` mode.

PAX learns a lot from other libraries too:
- PAX borrows the idea that _a module is also a pytree_ from [treex] and [equinox]. 
- PAX uses the concept of _trainable parameters_ and _non-trainable states_ from [dm-haiku].
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
| ``mnist_mixed_precision.py`` (experimental) | train an image classifier with mixed precision. |
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
