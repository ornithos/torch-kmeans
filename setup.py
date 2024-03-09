"""Bootstrap setup file."""
import os

from setuptools import find_packages, setup

# get the current package version
VERSION = "0.0.1"

# get the current path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# parse the readme into a variable
with open(os.path.join(CURRENT_PATH, "README.md"), "r", encoding="utf8") as rmd:
    long_desc = rmd.read()

# fetch the requirements required
with open(os.path.join(CURRENT_PATH, "requirements.txt"), "r", encoding="utf8") as req_file:
    requirements = req_file.read().split("\n")


# now create the setup tools script
setup(
    name="torch_kmeans",
    version=VERSION,
    author="Alex Bird",
    author_email="alex@quine.sh",
    description="KMeans clustering for PyTorch Tensors on CPU and GPU",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/ornithos/torch-kmeans",
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.10",
    install_requires=requirements,
)
