import msgpack
import copy

MSGPACK_ENCODERS = {}
MSGPACK_DECODERS = {}


def register_msgpack_encoder(tid, type):
    def _register_msgpack_encoder(fn):
        MSGPACK_ENCODERS[type] = (tid, fn)
        return fn
    return _register_msgpack_encoder


def register_msgpack_decoder(tid):
    def _register_msgpack_decoder(fn):
        MSGPACK_DECODERS[tid] = fn
        return fn
    return _register_msgpack_decoder


def to_msgpack_dict(obj):
    if obj.__class__ in MSGPACK_ENCODERS:
        tid, encode = MSGPACK_ENCODERS[obj.__class__]
        d = {'_tid': tid}
        d.update(encode(obj))
        return d
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_msgpack_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((to_msgpack_dict(k), to_msgpack_dict(v))
                          for k, v in obj.items())
    else:
        return copy.deepcopy(obj)


def from_msgpack_dict(d):
    if isinstance(d, dict):
        if '_tid' in d and d['_tid'] in MSGPACK_DECODERS:
            return MSGPACK_DECODERS[d['_tid']](d)
        else:
            return {k: from_msgpack_dict(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return type(d)(from_msgpack_dict(child) for child in d)
    else:
        return d


def to_msgpack(obj):
    return msgpack.packb(to_msgpack_dict(obj))


def from_msgpack(obj):
    return from_msgpack_dict(msgpack.unpackb(obj))
