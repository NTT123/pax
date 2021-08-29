"""Train a transformer language model on TPU (if available)."""

import inspect
import os
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import optax
import pax
import tensorflow as tf
from pax.optim import Optimizer
from tqdm.auto import tqdm

from model import Transformer

pax.seed_rng_key(42)


def setup_tpu_device():
    print("Setting up TPU cores")
    jax.tools.colab_tpu.setup_tpu()
    print(jax.devices())


if "COLAB_TPU_ADDR" in os.environ:
    # TPU config
    steps_per_update = 50
    num_devices = jax.device_count()
    batch_size = 32 * num_devices * steps_per_update
    seq_len = 128 + 1
    vocab_size = 256
    hidden_dim = 512
    num_steps = 50_000
    num_layers = 6
    setup_tpu_device()
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


class LM(pax.Module):
    """A RNN language model."""

    transformer: Transformer
    embed: pax.Module
    output: pax.Module

    vocab_size: int
    hidden_dim: int

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        """
        Arguments:
            vocab_size: int, size of the alphabet.
            hidden_dim: int, hidden dim.
            num_layers: int, num transformer blocks.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = pax.haiku.embed(vocab_size, hidden_dim)
        self.transformer = Transformer(hidden_dim, 8, num_layers, 0.1)
        self.output = pax.haiku.linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        logits = self.output(x)
        return logits

    def inference(self, prompt: List[int] = [], length=32):
        start_idx = len(prompt)
        pad_len = length - len(prompt)
        prompt = prompt + [0] * pad_len
        total_len = length

        @jax.jit
        def step(x):
            x = self.embed(x)
            x = self.transformer(x)
            logits = self.output(x)
            return logits

        for i in range(start_idx, total_len):
            x = jnp.array([prompt], dtype=jnp.int32)
            logits = step(x)
            x = jnp.argmax(logits[0, i], axis=-1)
            prompt[i] = x.item()
        return prompt


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


data = inspect.getsource(LM)  # a _true_ AGI learns about itself.
data_token = tokenize(data)
test_prompt = "class LM(pax.Module):"

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

losses = 0.0
tr = tqdm(range(0, 1 + num_steps, steps_per_update), desc="training")
for step in tr:
    batch = next(tfdata)
    # (num_devices,) is for jax.pmap, (steps_per_update,) is for jax.lax.scan
    batch = jnp.reshape(batch, (num_devices, steps_per_update, -1) + batch.shape[1:])
    loss, net, optimizer = update_fn(net, optimizer, batch)
    losses = losses + loss
    if step % 1000 == 0:
        loss = jnp.mean(losses) / (1000 if step > 0 else steps_per_update)
        losses = 0.0
        # eval on a single device
        eval_net = jax.tree_map(lambda x: x[0], net.eval())
        out = eval_net.inference(
            prompt=tokenize(test_prompt), length=(100 if step < num_steps else 1000)
        )
        text = detokenize(out)
        tr.write(
            f"[step {step}]  loss {loss:.3f}\nPrompt: {test_prompt}\n========\n{text}\n========"
        )
