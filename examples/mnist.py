"""train a handwritten digit classifier."""

import pickle
from functools import partial
from pathlib import Path
from typing import List, Mapping, Tuple

import jax
import jax.numpy as jnp
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
            conv = pax.nn.Conv2D((1 if i == 0 else 32), 32, 6, padding="VALID")
            bn = pax.nn.BatchNorm2D(32, True, True, 0.9)
            layers.append((conv, bn))
        self.layers = layers
        self.output = pax.nn.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray):
        for conv, bn in self.layers:
            x = bn(conv(x))
            x = jax.nn.relu(x)
        x = self.output(x)
        return jnp.squeeze(x, (1, 2))


def loss_fn(params: ConvNet, model: ConvNet, batch: Batch):
    model = model.update(params)
    x = batch["image"].astype(jnp.float32) / 255
    target = batch["label"]
    logits = model(x)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    log_pr = jnp.sum(jax.nn.one_hot(target, log_pr.shape[-1]) * log_pr, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, (loss, model)


@pax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = pax.enable_eval_mode(model)
    return loss_fn(pax.select_parameters(model), model, batch)[0]


@pax.jit
def update_fn(model: ConvNet, optimizer: GradientTransformation, batch: Batch):
    grads, (loss, model) = pax.grads_with_aux(model, fn=loss_fn, inputs=batch)
    model, optimizer = pax.apply_grads(model, optimizer, grads=grads)
    return model, optimizer, loss


net = ConvNet()
print(net.summary())
params = pax.select_parameters(net)
optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
)(params)


def load_dataset(split: str):
    """Loads the dataset as a tensorflow dataset."""
    ds = tfds.load("mnist:3.*.*", split=split)
    return ds


train_data = load_dataset("train").shuffle(10 * batch_size).batch(batch_size)
test_data = load_dataset("test").shuffle(10 * batch_size).batch(batch_size)


def save_ckpt(epoch: int, model: pax.Module, path: Path):
    model = jax.tree_map(lambda x: jax.device_get(x), model)
    leaves, treedef = jax.tree_flatten(model)
    del treedef
    with open(path, "wb") as f:
        pickle.dump({"epoch": epoch, "leaves": leaves}, f)


def load_ckpt(model, path: Path):
    """Load model from saved tree leaves"""
    leaves, treedef = jax.tree_flatten(model)
    with open(path, "rb") as f:
        dic = pickle.load(f)
    return dic["epoch"], jax.tree_unflatten(treedef, dic["leaves"])


# resume from the latest checkpoint
ckpts = sorted(Path("/tmp").glob("pax_mnist_ckpt_*.pickle"))
if len(ckpts) > 0:
    print("loading checkpoint at", ckpts[-1])
    last_epoch, net = load_ckpt(net, ckpts[-1])
else:
    last_epoch = -1

for epoch in range(last_epoch + 1, 10):
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

    save_ckpt(epoch, net, Path(f"/tmp/pax_mnist_ckpt_{epoch:02d}.pickle"))

    print(f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}")
