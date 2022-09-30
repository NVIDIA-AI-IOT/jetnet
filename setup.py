from importlib.metadata import entry_points
from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension
from glob import glob

ext_modules = [
    Pybind11Extension(
        "jetnet._jetnet_C",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        include_dirs=["src"]
    ),
]

setup(
    name="jetnet",
    version="0.0.0",
    description="Easy to use neural networks for NVIDIA Jetson (and desktop too!)",
    packages=find_packages(),
    entry_points={"console_scripts": ["jetnet = jetnet.__main__:main"]},
    install_requires=["pydantic"],
    ext_modules=ext_modules
)