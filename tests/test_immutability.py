import pytest
import pax


def test_immutability():
    f = pax.nn.Linear(3, 3)

    with pax.ctx.immutable():
        with pytest.raises(ValueError):
            f.c = 123

        f = f.freeze()
        f = f.unfreeze()
