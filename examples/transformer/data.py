import jax
import numpy as np
import tensorflow as tf


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
def double_buffer(ds, num_devices, steps_per_update):
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


def make_data_loader(data, seq_len, batch_size, num_devices, steps_per_update):
    data_token = tokenize(data)
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

    return double_buffer(tfdata, num_devices, steps_per_update)
