"""Commonly used utilities."""

import fnmatch
import importlib
import os
import inspect
from typing import List, Optional, Union, Any
from urllib.request import urlretrieve
from zipfile import ZipFile

from progressbar import ETA, Bar, ProgressBar, Timer


DATA_ROOT = "data"

__all__ = [
    "data_path",
    "download",
    "unzip",
    "parent_dir",
    "make_parent_dir",
    "import_object",
    "get_subclasses",
    "get_final_subclasses",
    "get_annotations",
    "get_public_annotations",
    "get_private_annotations"
]

def data_path(path: str):
    return os.path.join(DATA_ROOT, path)


def _pbar_widgets():
    return [" [", Timer(), "] ", Bar(), " (", ETA(), ") "]


class _DownloadProgressBar:
    def __init__(self):
        self._bar = None

    def __call__(self, block_index: int, block_size: int, total_size: int):

        if self._bar is None:
            self._bar = ProgressBar(
                maxval=total_size,
                widgets=_pbar_widgets(),
            )
            self._bar.start()

        downloaded = block_index * block_size

        if downloaded < total_size:
            self._bar.update(downloaded)
        else:
            self._bar.finish()
            self._bar = None


def download(url: str, path: Optional[str] = None) -> None:
    if path is None:
        path = os.path.join(os.curdir, os.path.basename(url))

    path = os.path.abspath(path)

    if os.path.isdir(path):
        raise IsADirectoryError("Path is a directory.")
    elif os.path.exists(path):
        raise FileExistsError("File already exists.")

    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError("Parent directory does not exist.")

    hook = _DownloadProgressBar()

    urlretrieve(url, path, reporthook=hook)


def unzip(
    file: str,
    path: Optional[str] = None,
    members: Union[List[str], None] = None,
    pattern: Optional[str] = None,
) -> None:
    with ZipFile(file) as zf:

        if members is None:
            members = zf.namelist()

        if pattern is not None:
            members = fnmatch.filter(members, pattern)

        bar = ProgressBar(
            maxval=len(members),
            widgets=_pbar_widgets(),
        )

        bar.start()

        for i, m in enumerate(members):
            zf.extract(m, path)
            bar.update(i)

        bar.finish()


def parent_dir(path: str) -> str:
    return os.path.dirname(os.path.abspath(path))

def make_parent_dir(path: str):
    if not os.path.exists(parent_dir(path)):
        os.makedirs(parent_dir(path))
        
def import_object(name: str) -> Any:
    """Import an attribute."""
    module_name = ".".join(name.split(".")[:-1])
    cls_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def get_subclasses(cls):
    subclasses = set()
    for subcls in cls.__subclasses__():
        if subcls not in subclasses:
            subclasses.add(subcls)
        subclasses = subclasses.union(get_subclasses(subcls))
    return subclasses


def get_final_subclasses(cls):
    return set(subcls for subcls in get_subclasses(cls) if not inspect.isabstract(subcls))


def get_annotations(cls):
    annotations = {}
    for base in cls.__bases__:
        annotations.update(get_annotations(base))
    if hasattr(cls, "__annotations__"):
        annotations.update(cls.__annotations__)
    return annotations


def get_public_annotations(cls):
    return {k: v for k, v in get_annotations(cls).items() if k[0] != '_'}


def get_private_annotations(cls):
    return {k: v for k, v in get_annotations(cls).items() if k[0] == '_'}



# def find_module_instances(module_str, _type):

#     module = importlib.import_module(module_str)

#     matches = []

#     for key in module.__dir__():
        
#         if key[0] == '_':
#             continue # don't include private members

#         value = getattr(module, key)

#         if not isinstance(value, _type):
#             continue
    
#         match = (module_str + "." + key, value)
#         matches.append(match)

#     return matches


# def find_module_subclasses(module_str, _type):

#     module = importlib.import_module(module_str)

#     matches = []

#     for key in module.__dir__():

#         if key[0] == '_':
#             continue # don't include private members

#         value = getattr(module, key)

#         if not issubclass(value, _type):
#             continue
    
#         match = (module_str + "." + key, value)
#         matches.append(match)

#     return matches

