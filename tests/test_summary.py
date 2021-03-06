import jax
import pax


def test_linear_summary():
    fc = pax.Linear(3, 3)
    assert fc.summary() == "Linear(in_dim=3, out_dim=3, with_bias=True)"


def test_sequential_summary():
    f = pax.Sequential(pax.Linear(3, 32), jax.nn.sigmoid, pax.Linear(32, 64))
    f1 = pax.BatchNorm1D(3)
    f1 = f1.set_attribute("T", f)
    print(f1.summary())
