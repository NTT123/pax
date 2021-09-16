import pytest
import pax


def test_immutability():
    f = pax.nn.Linear(3, 3)

    with pax.ctx.immutable():
        with pytest.raises(ValueError):
            f = f.eval()

        with pytest.raises(ValueError):
            f = f.train()

        with pytest.raises(ValueError):
            f = f.freeze()

        with pytest.raises(ValueError):
            f = f.unfreeze()
