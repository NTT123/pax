"""Train a rnn language model on TPU (if available)."""

import inspect
import os
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import opax
import pax
import tensorflow as tf
from tqdm.auto import tqdm

pax.seed_rng_key(42)


def setup_tpu_device():
    print("Setting up TPU cores")
    jax.tools.colab_tpu.setup_tpu()
    print(jax.devices())


if "COLAB_TPU_ADDR" in os.environ:
    # TPU config
    setup_tpu_device()
    steps_per_update = 50
    num_devices = jax.device_count()
    batch_size = 32 * num_devices * steps_per_update
    seq_len = 128
    vocab_size = 256
    hidden_dim = 512
    num_steps = 50_000
else:
    # CPU/GPU config
    steps_per_update = 1
    num_devices = jax.device_count()
    batch_size = 1 * num_devices * steps_per_update
    seq_len = 64
    vocab_size = 256
    hidden_dim = 256
    num_steps = 20_000


class LM(pax.Module):
    """A RNN language model."""

    lstm: pax.Module
    embed: pax.Module
    output: pax.Module

    vocab_size: int
    hidden_dim: int

    def __init__(self, vocab_size: int, hidden_dim: int):
        """
        Arguments:
            vocab_size: int, size of the alphabet.
            hidden_dim: int, number of LSTM cells.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = pax.nn.Embed(vocab_size, hidden_dim)
        self.lstm = pax.nn.LSTM(hidden_dim, hidden_dim)
        self.output = pax.nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embed(x)
        hx, x = pax.utils.scan(
            self.lstm,
            self.lstm.initial_state(x.shape[0]),
            x,
            time_major=False,
        )
        del hx
        logits = self.output(x)
        return logits

    def inference(self, prompt: List[int] = [], length=32):
        hx = self.lstm.initial_state(1)
        if len(prompt) == 0:
            prompt = [0]

        x = jnp.array([prompt[0]], dtype=jnp.int32)

        total_len = len(prompt) + length

        out = [x]

        @pax.jit
        def step(x, hx):
            x = self.embed(x)
            hx, x = self.lstm(hx, x)
            logits = self.output(x)
            return logits, hx

        for i in range(1, total_len):
            logits, hx = step(x, hx)
            if i >= len(prompt):
                x = jnp.argmax(logits, axis=-1)
            else:
                x = jnp.array([prompt[i]], dtype=jnp.int32)
            out.append(x)
        return jnp.concatenate(out)


def loss_fn(model: LM, batch: jnp.ndarray):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(targets, num_classes=model.vocab_size)
    loss = -jnp.mean(jnp.sum(targets * log_pr, axis=-1))
    return loss, (loss, model)


def update_step(model_and_optimizer, batch: jnp.ndarray):
    model, optimizer = model_and_optimizer
    grads, (loss, model) = pax.grad(loss_fn, has_aux=True)(model, batch)
    grads = jax.lax.pmean(grads, axis_name="i")
    model, optimizer = pax.apply_gradients(model, optimizer, grads=grads)
    return (model, optimizer), loss


@partial(pax.pmap, axis_name="i")
def update_fn(model, optimizer, multi_batch: jnp.ndarray):
    (model, optimizer), losses = pax.utils.scan(
        update_step, (model, optimizer), multi_batch
    )
    return model, optimizer, jnp.sum(losses)


net = LM(vocab_size=vocab_size, hidden_dim=hidden_dim)
optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.adam(1e-4),
)(net.parameters())

# replicate on multiple devices
net = jax.device_put_replicated(net, jax.devices())
print(net.summary())
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
        lambda x: tf.image.random_crop(x, [seq_len + 1]),
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
    # (num_devices,) is for pax.pmap, (steps_per_update,) is for pax.utils.scan
    batch = jnp.reshape(batch, (num_devices, steps_per_update, -1) + batch.shape[1:])
    net, optimizer, loss = update_fn(net, optimizer, batch)
    losses = losses + loss
    if step % 1000 == 0:
        loss = jnp.mean(losses) / (1000 if step > 0 else steps_per_update)
        losses = 0.0
        # eval on a single device
        eval_net = jax.tree_map(lambda x: x[0], net.eval())
        out = eval_net.inference(
            prompt=tokenize(test_prompt), length=(100 if step < num_steps else 1000)
        )
        text = detokenize(out.tolist())
        tr.write(
            f"[step {step}]  loss {loss:.3f}\nPrompt: {test_prompt}\n========\n{text}\n========"
        )
