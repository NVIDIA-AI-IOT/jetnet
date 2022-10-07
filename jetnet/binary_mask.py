import msgpack
import numpy as np
from abc import ABC, abstractmethod
from itertools import groupby
from jetnet.msgpack import (
    register_msgpack_decoder,
    register_msgpack_encoder
)
from jetnet.rle import rle_decode, rle_encode

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
    return rle_encode(obj.numpy())


@register_msgpack_decoder("BinaryMask")
def mask_from_msgpack_dict(msgpack_dict):
    return BinaryMask.from_numpy(rle_decode(msgpack_dict))