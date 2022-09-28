import pytest
import numpy as np
from jetnet.segmentation import Segmentation
from jetnet.msgpack import to_msgpack, from_msgpack


def test_binary_mask_msgpack():

    a = Segmentation.from_numpy(np.array([
        [0, 0, 1, 1],
        [2, 2, 2, 1],
        [0, 0, 3, 0]
    ], dtype=np.uint8))

    s = to_msgpack(a)

    b = from_msgpack(s)

    assert np.allclose(b.numpy(), a.numpy())