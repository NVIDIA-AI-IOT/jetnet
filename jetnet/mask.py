import msgpack
import numpy as np
from abc import ABC, abstractmethod
from itertools import groupby
from jetnet.msgpack import (
    register_msgpack_decoder,
    register_msgpack_encoder
)

def numpy_binary_mask_to_rle(mask):
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.flat)):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def numpy_binary_mask_from_rle(rle):
    mask = np.empty(rle['size'], dtype=np.uint8)
    idx = 0
    val = 0
    mask_flat = mask.flat
    for c in rle['counts']:
        mask_flat[idx:idx+c] = val
        val = not val
        idx += c
    return mask


class BinaryMask:

    def __init__(self, value):
        self._value = value
    
    def numpy(self):
        return self._value

    @classmethod
    def from_numpy(cls, value):
        return BinaryMask(value)


@register_msgpack_encoder("BinaryMask", BinaryMask)
def mask_to_msgpack_dict(obj: BinaryMask):
    return numpy_binary_mask_to_rle(obj.numpy())


@register_msgpack_decoder("BinaryMask")
def mask_from_msgpack_dict(msgpack_dict):
    return BinaryMask.from_numpy(numpy_binary_mask_from_rle(msgpack_dict))