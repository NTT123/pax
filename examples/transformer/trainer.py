"""Train a transformer language model on TPU (if available)."""

import inspect
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import optax
import pax
import tensorflow as tf
from pax.optim import Optimizer
from tqdm.auto import tqdm

from model import LM


def setup_tpu_device():
    print("Setting up TPU cores")
    jax.tools.colab_tpu.setup_tpu()
    print(jax.devices())


if "COLAB_TPU_ADDR" in os.environ:
    # TPU config
    # need to config TPU cores _before_ calling `jax.device_count`.
    setup_tpu_device()
    steps_per_update = 50
    num_devices = jax.device_count()
    batch_size = 32 * num_devices * steps_per_update
    seq_len = 256 + 1
    vocab_size = 256
    hidden_dim = 512
    num_steps = 1_000
    num_layers = 6
else:
    # CPU/GPU config
    steps_per_update = 1
    num_devices = jax.device_count()
    batch_size = 1 * num_devices * steps_per_update
    seq_len = 64 + 1
    vocab_size = 256
    hidden_dim = 256
    num_steps = 20_000
    num_layers = 2

pax.seed_rng_key(42)


def loss_fn(params: LM, model: LM, batch: jnp.ndarray):
    model = model.update(params)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(targets, num_classes=model.vocab_size)
    loss = -jnp.mean(jnp.sum(targets * log_pr, axis=-1))
    return loss, (loss, model)


def update_step(prev, batch: jnp.ndarray):
    model, optimizer = prev
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(
        model.parameters(), model, batch
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    model = optimizer.step(grads, model)
    return (model, optimizer), loss


@partial(jax.pmap, axis_name="i")
def update_fn(model: LM, optimizer: Optimizer, multi_batch: jnp.ndarray):
    (model, optimizer), losses = jax.lax.scan(
        update_step, (model, optimizer), multi_batch
    )
    return jnp.sum(losses), model, optimizer


net = LM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
optimizer = pax.optim.from_optax(
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
)(net.parameters())

# replicate on multiple devices
net = jax.device_put_replicated(net, jax.devices())
optimizer = jax.device_put_replicated(optimizer, jax.devices())


def tokenize(text):
    t = [0] + [ord(c) for c in text]  # ASCII, 0 is the [START] token
    return t


def detokenize(tokens):
    text = [chr(t) if t != 0 else "[START]" for t in tokens]
    return "".join(text)


def _device_put_sharded(sharded_tree, devices):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.api.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], devices
    )


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
    optimizer = pax.optim.from_optax(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
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
            lambda x: tf.image.random_crop(x, [seq_len]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    tfdata = double_buffer(tfdata)
    losses = 0.0
    tr = tqdm(range(0, 1 + num_steps, steps_per_update), desc="training")
    for step in tr:
        batch = next(tfdata)
        # (num_devices,) is for jax.pmap, (steps_per_update,) is for jax.lax.scan
        loss, net, optimizer = update_fn(net, optimizer, batch)
        losses = losses + loss
        if step % 1000 == 0:
            loss = jnp.mean(losses) / (1000 if step > 0 else steps_per_update)
            losses = 0.0
            # eval on a single device
            eval_net = jax.tree_map(lambda x: x[0], net.eval())
            out = eval_net.inference(
                prompt=tokenize(test_prompt),
                length=(128 if step < num_steps else 1024),
                train_seq_len=seq_len - 1,
            )
            text = detokenize(out)
            tr.write(
                f"[step {step}]  loss {loss:.3f}\nPrompt: {test_prompt}\n========\n{text}\n========"
            )


train()
