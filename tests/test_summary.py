import jax
import pax


def test_linear_summary():
    fc = pax.nn.Linear(3, 3)
    assert fc.summary() == "Linear[in_dim=3, out_dim=3, with_bias=True]"


def test_sequential_summary():
    f = pax.nn.Sequential(pax.nn.Linear(3, 32), jax.nn.sigmoid, pax.nn.Linear(32, 64))
    f1 = pax.nn.Linear(5, 5)

    @pax.pure
    def _mutate(f1):
        f1.T = f
        return f1

    f1 = _mutate(f1)

    print(f1.summary())
