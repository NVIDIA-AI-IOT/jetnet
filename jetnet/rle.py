import numpy as np
import jetnet._jetnet_C


def rle_encode(seg):
    rle = {'size': list(seg.shape)}
    counts, values = jetnet._jetnet_C.rle(seg)
    rle['counts'] = counts
    rle['values'] = values
    return rle


def rle_decode(rle):
    seg = np.empty(rle['size'], dtype=np.uint8)
    idx = 0
    seg_flat = seg.flat
    for c, v in zip(rle['counts'], rle['values']):
        seg_flat[idx:idx+c] = v
        idx += c
    return seg
