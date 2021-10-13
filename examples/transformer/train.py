"""Train a transformer language model on TPU (if available)."""

import inspect
import os
from functools import partial
from math import gamma
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import opax
import pax
import tensorflow as tf
from opax.transform import GradientTransformation
from tqdm.auto import tqdm

from model import LM


def setup_tpu_device():
    print("Setting up TPU cores")
    jax.tools.colab_tpu.setup_tpu()
    print(jax.devices())


# shared config
dropout = 0.1
learning_rate = 1e-4
vocab_size = 256
pax.seed_rng_key(42)

if "COLAB_TPU_ADDR" in os.environ:
    # TPU config
    # need to config TPU cores _before_ calling `jax.device_count`.
    setup_tpu_device()
    steps_per_update = 50
    num_devices = jax.device_count()
    batch_size = 32 * num_devices * steps_per_update
    seq_len = 256
    hidden_dim = 512
    num_steps = 1_000
    num_layers = 6
else:
    # CPU/GPU config
    steps_per_update = 1
    num_devices = jax.device_count()
    batch_size = 8 * num_devices * steps_per_update
    seq_len = 64
    hidden_dim = 256
    num_steps = 20_000
    num_layers = 2


def loss_fn(model: LM, batch: jnp.ndarray):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    model, logits = pax.module_and_value(model)(inputs)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(targets, num_classes=model.vocab_size)
    loss = -jnp.mean(jnp.sum(targets * log_pr, axis=-1))
    return loss, (loss, model)


def update_step(prev, batch: jnp.ndarray):
    model, optimizer = prev
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True, allow_int=True)(model, batch)
    grads = jax.lax.pmean(grads.parameters(), axis_name="i")
    model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
    return (model, optimizer), loss


@partial(jax.pmap, axis_name="i")
def update_fn(model: LM, optimizer: GradientTransformation, multi_batch: jnp.ndarray):
    (model, optimizer), losses = pax.scan(update_step, (model, optimizer), multi_batch)
    return model, optimizer, jnp.sum(losses)


def tokenize(text):
    t = [0] + [ord(c) for c in text]  # ASCII, 0 is the [START] token
    return t


def detokenize(tokens):
    text = [chr(t) if t != 0 else "[START]" for t in tokens]
    return "".join(text)


def _device_put_sharded(sharded_tree, devices):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], devices
    )


# Source: https://github.com/deepmind/dm-haiku/blob/8fad8c7503c5f56fa9ea9b53f71b7082704e3a3e/examples/imagenet/dataset.py#L163
def double_buffer(ds):
    """Keeps at least two batches on the accelerator.
    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.
    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.
    Args:
      ds: Iterable of batches of numpy arrays.
    Yields:
      Batches of sharded device arrays.
    """
    batch = None
    devices = jax.devices()
    for next_batch in ds:
        assert next_batch is not None
        next_batch = np.reshape(
            next_batch, (num_devices, steps_per_update, -1) + next_batch.shape[1:]
        )
        next_batch = _device_put_sharded(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


def train():
    net = LM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
    print(net.summary())
    optimizer = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.adam(learning_rate),
    )(net.parameters())

    # replicate on multiple devices
    net = jax.device_put_replicated(net, jax.devices())
    optimizer = jax.device_put_replicated(optimizer, jax.devices())

    data = inspect.getsource(LM)  # a _true_ AGI learns about itself.
    data_token = tokenize(data)
    test_prompt = data[:20]
    data_token = [0] * seq_len + data_token

    tfdata = (
        tf.data.Dataset.from_tensors(data_token)
        .repeat()
        .map(
            lambda x: tf.image.random_crop(x, [seq_len + 1]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    tfdata = double_buffer(tfdata)
    total_losses = 0.0
    tr = tqdm(range(0, 1 + num_steps, steps_per_update), desc="training")
    for step in tr:
        batch = next(tfdata)
        # (num_devices,) is for jax.pmap, (steps_per_update,) is for pax.scan
        net, optimizer, loss = update_fn(net, optimizer, batch)
        total_losses = total_losses + loss
        if step % 1000 == 0:
            loss = jnp.mean(total_losses) / (1000 if step > 0 else steps_per_update)
            total_losses = jnp.zeros_like(total_losses)
            # eval on a single device
            eval_net = jax.tree_map(lambda x: x[0], net.eval())
            out = eval_net.inference(
                prompt=tokenize(test_prompt),
                length=(128 if step < num_steps else 1024),
                train_seq_len=seq_len,
            )
            text = detokenize(out)
            tr.write(
                f"[step {step}]  loss {loss:.3f}\nPrompt: {test_prompt}\n========\n{text}\n========"
            )


if __name__ == "__main__":
    train()
