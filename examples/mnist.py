"""train a handwritten digit classifier."""

import pickle
from pathlib import Path
from typing import List, Mapping

import jax
import jax.numpy as jnp
import optax
import pax
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

Batch = Mapping[str, jnp.ndarray]

# config
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
weight_decay = 1e-4


class ConvNet(pax.Module):
    """ConvNet module."""

    convs: List[pax.nn.Conv2D] = None
    bns: List[pax.nn.BatchNorm] = None
    output: pax.nn.Conv2D

    def __init__(self):
        super().__init__()
        self.register_module_subtree(
            "convs",
            [
                pax.nn.Conv2D((1 if i == 0 else 32), 32, 6, padding="VALID")
                for i in range(5)
            ],
        )
        self.register_module_subtree(
            "bns", [pax.haiku.batch_norm_2d(32) for _ in range(5)]
        )
        self.output = pax.nn.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray):
        for conv, bn in zip(self.convs, self.bns):
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


@jax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = model.eval()
    return loss_fn(model.parameters(), model, batch)[0]


@jax.jit
def update_fn(model: ConvNet, optimizer: pax.Optimizer, batch: Batch):
    params = model.parameters()
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(params, model, batch)
    model = optimizer.step(grads, model)
    return loss, model, optimizer


net = ConvNet()
print(net.summary())
optimizer = pax.optim.from_optax(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
)(net.parameters())


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
        loss, net, optimizer = update_fn(net, optimizer, batch)
        losses = losses + loss
    loss = losses / len(train_data)

    test_losses = 0.0
    for batch in tqdm(test_data, desc="eval", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        test_losses = test_losses + test_loss_fn(net, batch)
    test_loss = test_losses / len(test_data)

    save_ckpt(epoch, net, Path(f"/tmp/pax_mnist_ckpt_{epoch:02d}.pickle"))

    print(f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}")
