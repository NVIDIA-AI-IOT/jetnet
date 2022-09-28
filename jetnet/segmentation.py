
import msgpack
import numpy as np
from abc import ABC, abstractmethod
from itertools import groupby
from jetnet.msgpack import (
    register_msgpack_decoder,
    register_msgpack_encoder
)


def numpy_segmentation_to_rle(seg):
    rle = {'counts': [], 'values': [], 'size': list(seg.shape)}
    counts = rle.get('counts')
    values = rle.get('values')
    for i, (value, elements) in enumerate(groupby(seg.flat)):
        values.append(int(value))
        counts.append(len(list(elements)))
    return rle


def numpy_segmentation_from_rle(rle):
    seg = np.empty(rle['size'], dtype=np.uint8)
    idx = 0
    seg_flat = seg.flat
    for c, v in zip(rle['counts'], rle['values']):
        seg_flat[idx:idx+c] = v
        idx += c
    return seg


class Segmentation:

    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value

    @classmethod
    def from_numpy(cls, value):
        return Segmentation(value)


@register_msgpack_encoder("Segmentation", Segmentation)
def segmentation_to_msgpack_dict(obj: Segmentation):
    return numpy_segmentation_to_rle(obj.numpy())


@register_msgpack_decoder("Segmentation")
def segmentation_from_msgpack_dict(msgpack_dict):
    return Segmentation.from_numpy(numpy_segmentation_from_rle(msgpack_dict))