import pytest
import numpy as np
from jetnet.mask import BinaryMask
from jetnet.msgpack import to_msgpack, from_msgpack


def test_binary_mask_msgpack():

    a = BinaryMask.from_numpy(np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]))

    s = to_msgpack(a)

    b = from_msgpack(s)

    assert np.allclose(b.numpy(), a.numpy())