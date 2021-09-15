"""train a handwritten digit classifier."""

from typing import List, Mapping, Tuple

import jax
import jax.numpy as jnp
import jmp
import opax
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

        self.register_module_subtree("layers", layers)
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
    model = model.eval()
    return loss_fn(model.parameters(), model, batch)[0]


@pax.jit
def update_fn(model: ConvNet, optimizer: pax.Module, batch: Batch):
    params = model.parameters()
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(params, model, batch)
    model = model.update(
        optimizer.step(grads, model.parameters()),
    )
    return loss, model, optimizer


net = ConvNet()


# TODO: check why this makes training so slow on CPU.
half = jnp.float16  # or bfloat16
full = jnp.float32
linear_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
batchnorm_policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)


def mp_policy_fn(mod):
    if isinstance(mod, pax.nn.Conv2D):
        return mod.mixed_precision(linear_policy)
    elif mod.__class__.__name__.startswith("BatchNorm"):
        return mod.mixed_precision(batchnorm_policy)
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
        loss, net, optimizer = update_fn(net, optimizer, batch)
        losses = losses + loss
    loss = losses / len(train_data)

    test_losses = 0.0
    for batch in tqdm(test_data, desc="eval", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        test_losses = test_losses + test_loss_fn(net, batch)
    test_loss = test_losses / len(test_data)

    print(f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}")
