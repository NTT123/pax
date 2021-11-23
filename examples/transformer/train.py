"""Train a transformer language model on TPU (if available)."""

import inspect
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import opax
import pax
from opax import GradientTransformation
from tqdm.auto import tqdm

from data import detokenize, make_data_loader, tokenize
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

    model, logits = pax.purecall(model, inputs)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(targets, num_classes=model.vocab_size)
    loss = -jnp.mean(jnp.sum(targets * log_pr, axis=-1))
    return loss, model


def update_step(model_and_optim: Tuple[LM, GradientTransformation], batch: jnp.ndarray):
    model, optimizer = model_and_optim
    (loss, model), grads = pax.value_and_grad(loss_fn, has_aux=True)(model, batch)
    grads = jax.lax.pmean(grads, axis_name="i")
    params = model.parameters()
    optimizer, updates = pax.purecall(optimizer, grads, params)
    params = params.map(jax.lax.sub, updates)
    model = model.update_parameters(params)
    return (model, optimizer), loss


@partial(jax.pmap, axis_name="i")
def update_fn(model: LM, optimizer: GradientTransformation, multi_batch: jnp.ndarray):
    (model, optimizer), losses = pax.scan(update_step, (model, optimizer), multi_batch)
    return model, optimizer, jnp.sum(losses)


def train():
    net = LM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
    print(net.summary())
    optimizer = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.adam(learning_rate),
    ).init(net.parameters())

    data = inspect.getsource(LM)  # a _true_ AGI learns about itself.
    test_prompt = data[:20]
    data_iter = make_data_loader(
        data,
        seq_len=seq_len,
        batch_size=batch_size,
        num_devices=num_devices,
        steps_per_update=steps_per_update,
    )

    # replicate on multiple devices
    net = jax.device_put_replicated(net, jax.devices())
    optimizer = jax.device_put_replicated(optimizer, jax.devices())

    total_losses = 0.0
    tr = tqdm(range(0, 1 + num_steps, steps_per_update), desc="training")
    for step in tr:
        batch = next(data_iter)
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
                f"[step {step}]  loss {loss:.3f}\n"
                f"Prompt: {test_prompt}\n"
                f"========\n"
                f"{text}\n"
                f"========"
            )


if __name__ == "__main__":
    train()
