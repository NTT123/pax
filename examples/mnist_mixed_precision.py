"""train a handwritten digit classifier."""

from functools import partial
from typing import List, Mapping, Tuple

import jax
import jax.numpy as jnp
import jmp
import opax
import pax
import tensorflow_datasets as tfds
from opax.transform import GradientTransformation
from tqdm.auto import tqdm

Batch = Mapping[str, jnp.ndarray]

# config
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
weight_decay = 1e-4
pax.seed_rng_key(42)


class ConvNet(pax.Module):
    """ConvNet module."""

    layers: List[Tuple[pax.nn.Conv2D, pax.nn.BatchNorm2D]]
    output: pax.nn.Conv2D

    def __init__(self):
        super().__init__()
        layers = []
        for i in range(5):
            conv_in = 1 if i == 0 else 32
            conv = pax.nn.Conv2D(conv_in, 32, 6, padding="VALID")
            bn = pax.nn.BatchNorm2D(32)
            layers.append((conv, bn))

        self.layers = layers
        self.output = pax.nn.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray):
        for conv, bn in self.layers:
            x = bn(conv(x))
            x = jax.nn.relu(x)
        x = self.output(x)
        return jnp.squeeze(x, (1, 2))


@pax.pure
def loss_fn(model: ConvNet, batch: Batch):
    x = batch["image"].astype(jnp.float32) / 255
    target = batch["label"]
    logits = model(x)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    log_pr = jnp.sum(jax.nn.one_hot(target, log_pr.shape[-1]) * log_pr, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, (loss, model)


@jax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = model.eval()
    return loss_fn(model, batch)[0]


@jax.jit
def update_fn(model: ConvNet, optimizer: GradientTransformation, batch: Batch):
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(model, batch)
    model, optimizer = pax.apply_gradients(model, optimizer, grads=grads)
    return model, optimizer, loss


net = ConvNet()


# TODO: check why this makes training so slow on CPU.
half = jmp.half_dtype()
full = jnp.float32
linear_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
batchnorm_policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)


def mp_policy_fn(mod):
    if isinstance(mod, pax.nn.Conv2D):
        return pax.apply_mp_policy(mod, mp_policy=linear_policy)
    elif mod.__class__.__name__.startswith("BatchNorm"):
        return pax.apply_mp_policy(mod, mp_policy=batchnorm_policy)
    else:
        # unchanged
        return mod


net = net.apply(mp_policy_fn)

print(net.summary())
optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
)(net.parameters())


def load_dataset(split: str):
    """Loads the dataset as a tensorflow dataset."""
    ds = tfds.load("mnist:3.*.*", split=split)
    return ds


train_data = load_dataset("train").shuffle(10 * batch_size).batch(batch_size)
test_data = load_dataset("test").shuffle(10 * batch_size).batch(batch_size)


for epoch in range(0, 10):
    losses = 0.0
    for batch in tqdm(train_data, desc="train", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        net, optimizer, loss = update_fn(net, optimizer, batch)
        losses = losses + loss
    loss = losses / len(train_data)

    test_losses = 0.0
    for batch in tqdm(test_data, desc="eval", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        test_losses = test_losses + test_loss_fn(net, batch)
    test_loss = test_losses / len(test_data)

    print(f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}")
