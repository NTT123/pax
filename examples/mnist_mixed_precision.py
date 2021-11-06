"""train a handwritten digit classifier with mixed precision."""

from typing import List, Mapping, Tuple

import fire
import jax
import jax.numpy as jnp
import jmp
import opax
import pax
import tensorflow_datasets as tfds
from opax.transform import GradientTransformation
from tqdm.auto import tqdm

Batch = Mapping[str, jnp.ndarray]


class ConvNet(pax.Module):
    """ConvNet module."""

    layers: List[Tuple[pax.nn.Conv2D, pax.nn.BatchNorm2D]]
    output: pax.nn.Conv2D

    def __init__(self):
        super().__init__()
        self.layers = []
        for i in range(5):
            conv_in = 1 if i == 0 else 32
            conv = pax.nn.Conv2D(conv_in, 32, 6, padding="VALID")
            bn = pax.nn.BatchNorm2D(32)
            self.layers.append((conv, bn))

        self.output = pax.nn.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray):
        for conv, bn in self.layers:
            x = bn(conv(x))
            x = jax.nn.relu(x)
        x = self.output(x)
        return jnp.squeeze(x, (1, 2))


def loss_fn(model: ConvNet, batch: Batch, loss_scale: jmp.LossScale):
    x = batch["image"].astype(jnp.float32) / 255
    target = batch["label"]
    model, logits = pax.module_and_value(model)(x)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    log_pr = jnp.sum(jax.nn.one_hot(target, log_pr.shape[-1]) * log_pr, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss_scale.scale(loss), (loss, model)


@jax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = model.eval()
    return loss_fn(model, batch, jmp.NoOpLossScale())[0]


def apply_gradients_w_loss_scale(
    model: pax.Module,
    optimizer: opax.GradientTransformation,
    loss_scale: jmp.LossScale,
    grads: pax.Module,
):
    grads = loss_scale.unscale(grads)
    skip_nonfinite_updates = isinstance(loss_scale, jmp.DynamicLossScale)
    if skip_nonfinite_updates:
        grads_finite = jmp.all_finite(grads)
        loss_scale = loss_scale.adjust(grads_finite)
        model, optimizer = opax.apply_gradients(
            model, optimizer, grads=grads, all_finite=grads_finite
        )
    else:
        model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
    return model, optimizer, loss_scale


@jax.jit
def update_fn(
    model: ConvNet,
    optimizer: GradientTransformation,
    loss_scale: jmp.LossScale,
    batch: Batch,
):
    grad_fn = pax.grad(loss_fn, has_aux=True)
    grads, (loss, model) = grad_fn(model, batch, loss_scale=loss_scale)
    return apply_gradients_w_loss_scale(model, optimizer, loss_scale, grads) + (loss,)


def load_dataset(split: str):
    """Loads the dataset as a tensorflow dataset."""
    ds = tfds.load("mnist:3.*.*", split=split)
    return ds


def mp_policy_fn(mod):
    half = jmp.half_dtype()
    full = jnp.float32
    linear_policy = jmp.Policy(compute_dtype=half, param_dtype=full, output_dtype=full)
    bn_policy = jmp.Policy(compute_dtype=full, param_dtype=full, output_dtype=full)

    if isinstance(mod, pax.nn.Conv2D):
        return pax.apply_mp_policy(mod, mp_policy=linear_policy)
    elif isinstance(mod, pax.nn.BatchNorm2D):
        return pax.apply_mp_policy(mod, mp_policy=bn_policy)
    else:
        return mod  # unchanged


def train(batch_size=32, num_epochs=5, learning_rate=1e-4, weight_decay=1e-4):
    pax.seed_rng_key(42)

    net = ConvNet()
    net = net.apply(mp_policy_fn)
    print(net.summary())
    optimizer = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )(net.parameters())

    loss_scale = jmp.DynamicLossScale(jmp.half_dtype()(2 ** 15), period=2000)

    train_data = (
        load_dataset("train")
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
    )
    test_data = load_dataset("test").batch(batch_size, drop_remainder=True)

    for epoch in range(0, num_epochs):
        losses = 0.0
        for batch in tqdm(train_data, desc="train", leave=False):
            batch = jax.tree_map(lambda x: x.numpy(), batch)
            net, optimizer, loss_scale, loss = update_fn(
                net, optimizer, loss_scale, batch
            )
            losses = losses + loss
        loss = losses / len(train_data)

        test_losses = 0.0
        for batch in tqdm(test_data, desc="eval", leave=False):
            batch = jax.tree_map(lambda x: x.numpy(), batch)
            test_losses = test_losses + test_loss_fn(net, batch)
        test_loss = test_losses / len(test_data)

        print(
            f"[Epoch {epoch}]  train loss {loss:.3f}  test loss"
            f" {test_loss:.3f}  loss scale {loss_scale.loss_scale}"
        )


if __name__ == "__main__":
    fire.Fire(train)
