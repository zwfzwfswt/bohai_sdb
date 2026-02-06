#!/usr/bin/env python3
"""Setup configuration for bohai_sdb package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bohai_sdb",
    version="1.0.0",
    author="zwfzwfswt",
    author_email="wtshang@yic.ac.cn",
    description="Bohai Sea Seabed Database - A database system for managing seabed engineering geology data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zwfzwfswt/bohai_sdb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        # SQLite3 is included with Python standard library
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bohai-sdb-demo=bohai_sdb:main",
        ],
    },
)
