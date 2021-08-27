"""Train a rnn language model."""

import jax
import jax.numpy as jnp
import optax
import pax


class LM(pax.Module):
    """A RNN language model."""

    lstm: pax.Module
    embed: pax.Module
    output: pax.Module

    def __init__(self, hidden_dim: int):
        self.embed = pax.haiku.embed(256, hidden_dim)
        self.lstm = pax.haiku.lstm(hidden_dim)
        self.output = pax.haiku.linear(hidden_dim, 256)

    def __call__(self, x):
        x = self.embed(x)
        x, hx = pax.haiku.dynamic_unroll(
            self.lstm, x, self.lstm.initial_state(x.shape[0]), time_major=False
        )
        del hx
        logits = self.output(x)
        return logits

    @jax.jit
    def inference(self):
        hx = self.lstm.initial_state(1)
        x = jnp.zeros((1,), dtype=jnp.int32)

        out = [x]
        for i in range(32):
            x = self.embed(x)
            x, hx = self.lstm(x, hx)
            logits = self.output(x)
            x = jnp.argmax(logits, axis=-1)
            out.append(x)
        return jnp.concatenate(out)


def loss_fn(params: LM, model: LM, batch: jnp.ndarray):
    model = model.update(params)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    targets = jax.nn.one_hot(targets, num_classes=log_pr.shape[-1])
    loss = -jnp.mean(jnp.sum(targets * log_pr, axis=-1))
    return loss, (loss, model)


@jax.jit
def update_fn(model: LM, optimizer: pax.optim.Optimizer, batch: jnp.ndarray):
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True)(
        model.parameters(), model, batch
    )
    model = optimizer.step(grads, model)
    return loss, model, optimizer


net = LM(64)
optimizer = pax.optim.from_optax(optax.adam(1e-4))(net.parameters())
x = jnp.arange(0, 33, dtype=jnp.int32)[None]

losses = 0.0
for step in range(1, 1 + 1_000):
    loss, net, optimizer = update_fn(net, optimizer, x)
    losses = losses + loss
    if step % 100 == 0:
        loss = losses / 100
        losses = 0.0
        net = net.eval()
        out = net.inference()
        print(f"[step {step}]  loss {loss:.3f}  sample {out.tolist()}")
        net = net.train()
