#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="figaroh",
    version="0.0.0",
    license="BSD-2-Clause",
    description="",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
        "Private :: Do Not Upload",
    ],
    keywords=[],
    python_requires=">=3.8",
    install_requires=[
        "pin",  # pinocchio
        "typer",
        "numdifftools",
        "quadprog",
        "numpy",
        "scipy",
        "picos",
        "pandas",
        "meshcat",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest"]
    },
    entry_points={
        "console_scripts": [
            "figaroh = figaroh.cli:main",
        ]
    },
)
