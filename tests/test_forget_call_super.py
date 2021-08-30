import pax
import pytest


def test_forget_call_super_at_init():
    class M(pax.Module):
        def __init__(self):
            self.fc = pax.nn.Linear(3, 3)

    with pytest.raises(RuntimeError):
        m = M()
