"""
Test PAX in multithread environment.
"""

import queue
import threading
import time

import jax.numpy as jnp
import pax


class DelayedCounter(pax.Module):
    def __init__(self):
        super().__init__()
        self.counter = jnp.array(0)

    def __call__(self):
        time.sleep(1)
        self.counter += 1
        time.sleep(1)
        return self.counter


def test_multithread():
    @pax.pure
    def update(c: DelayedCounter, q):
        o = c()
        q.put(o)

    c1 = DelayedCounter()
    c2 = DelayedCounter()
    q = queue.Queue()
    x = threading.Thread(target=update, args=(c1, q))
    y = threading.Thread(target=update, args=(c2, q))
    x.start()
    y.start()
    x.join()
    y.join()
    q.get(timeout=1)
    q.get(timeout=1)
