"""Setup file for package."""

import pathlib
from setuptools import find_packages, setup

long_description = (pathlib.Path(__file__).parent.resolve() / "README.md").read_text(
    encoding="utf-8"
)

setup(
    name="python-som",
    version="0.0.1a2",
    description="Python implementation of the Self-Organizing Map",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andremsouza/python-som",
    author="André Moreira Souza",
    author_email="msouza.andre@hotmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="som kohonen-map self-organizing-map machine-learning",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=["numpy", "pandas", "scikit-learn"],
    license="MIT",
    project_urls={
        "Bug Reports": "https://github.com/andremsouza/python-som/issues",
        "Source": "https://github.com/andremsouza/python-som",
    },
)
