from importlib.metadata import entry_points
from setuptools import find_packages, setup

setup(
    name="jetnet",
    version="0.0.0",
    description="Easy to use neural networks for NVIDIA Jetson (and desktop too!)",
    packages=find_packages(),
    entry_points={"console_scripts": ["jetnet = jetnet.__main__:main"]},
    install_requires=["pydantic"]
)