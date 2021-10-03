import jax
import pax


def test_linear_summary():
    fc = pax.nn.Linear(3, 3)
    assert fc.summary() == "Linear[in_dim=3, out_dim=3, with_bias=True]"


def test_sequential_summary():
    f = pax.nn.Sequential(pax.nn.Linear(3, 32), jax.nn.sigmoid, pax.nn.Linear(32, 64))
    f1 = pax.nn.Linear(5, 5)

    def add_T(m):
        m.T = f
        return m

    f1 = pax.mutate(f1, with_fn=add_T)
    print(f1.summary())
