"""train a handwritten digit classifier."""

import pickle
from pathlib import Path
from typing import Mapping

import fire
import jax
import jax.numpy as jnp
import opax
import pax
import tensorflow_datasets as tfds
from opax import GradientTransformation
from tqdm.auto import tqdm

Batch = Mapping[str, jnp.ndarray]


class ConvNet(pax.Module):
    """ConvNet module."""

    layers: pax.Sequential

    def __init__(self):
        super().__init__()
        self.layers = pax.Sequential()
        for i in range(5):
            self.layers >>= pax.Conv2D((1 if i == 0 else 32), 32, 6, padding="VALID")
            self.layers >>= pax.BatchNorm2D(32, True, True, 0.9)
            self.layers >>= jax.nn.relu
        self.layers >>= pax.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layers(x)
        return jnp.squeeze(x, (1, 2))


def loss_fn(model: ConvNet, batch: Batch):
    x = batch["image"].astype(jnp.float32) / 255
    target = batch["label"]
    model, logits = pax.purecall(model, x)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    log_pr = jnp.sum(jax.nn.one_hot(target, log_pr.shape[-1]) * log_pr, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, model


@jax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = model.eval()
    return loss_fn(model, batch)[0]


@jax.jit
def update_fn(model: ConvNet, optimizer: GradientTransformation, batch: Batch):
    (loss, model), grads = pax.value_and_grad(loss_fn, has_aux=True)(model, batch)
    params = model.parameters()
    optimizer, updates = pax.purecall(optimizer, grads, params)
    params = params.map(jax.lax.sub, updates)
    model = model.update_parameters(params)
    return model, optimizer, loss


def load_dataset(split: str):
    """Loads the dataset as a tensorflow dataset."""
    ds = tfds.load("mnist:3.*.*", split=split)
    return ds


def save_ckpt(epoch: int, model: ConvNet, path: Path):
    model = jax.device_get(model)
    with open(path, "wb") as f:
        pickle.dump({"epoch": epoch, "state_dict": model.state_dict()}, f)


def load_ckpt(model: ConvNet, path: Path):
    """Load model from saved tree leaves"""
    with open(path, "rb") as f:
        dic = pickle.load(f)
    return dic["epoch"], model.load_state_dict(dic["state_dict"])


def train(
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    ckpt_dir="/tmp",
):
    pax.seed_rng_key(42)

    # model
    net = ConvNet()
    print(net.summary())

    # optimizer
    optimizer = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    ).init(net.parameters())

    # data
    train_data = load_dataset("train").shuffle(10 * batch_size).batch(batch_size)
    test_data = load_dataset("test").shuffle(10 * batch_size).batch(batch_size)

    # resume from the latest checkpoint
    ckpts = sorted(Path(ckpt_dir).glob("pax_mnist_ckpt_*.pickle"))
    if len(ckpts) > 0:
        print("loading checkpoint at", ckpts[-1])
        last_epoch, net = load_ckpt(net, ckpts[-1])
    else:
        last_epoch = -1

    # training loop
    for epoch in range(last_epoch + 1, num_epochs):
        losses = 0.0

        # training
        for batch in tqdm(train_data, desc="train", leave=False):
            batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
            net, optimizer, loss = update_fn(net, optimizer, batch)
            losses = losses + loss
        loss = losses / len(train_data)

        # testing
        test_losses = 0.0
        for batch in tqdm(test_data, desc="test", leave=False):
            batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
            test_losses = test_losses + test_loss_fn(net, batch)
        test_loss = test_losses / len(test_data)

        save_ckpt(epoch, net, Path(ckpt_dir) / f"pax_mnist_ckpt_{epoch:02d}.pickle")
        # logging
        print(f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}")


if __name__ == "__main__":
    fire.Fire(train)
