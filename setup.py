#!/usr/bin/env python3
"""
RustTorch - High-Performance PyTorch Extension in Rust

This setup.py provides a convenient way to install RustTorch alongside PyTorch.
For development, use maturin directly in the rusttorch-py directory.
"""

from setuptools import setup
import subprocess
import sys
import os

def build_rust_extension():
    """Build the Rust extension using maturin"""
    rusttorch_py_dir = os.path.join(os.path.dirname(__file__), "rusttorch-py")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "maturin"],
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["maturin", "build", "--release"],
            cwd=rusttorch_py_dir,
        )
        print("✓ RustTorch extension built successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build RustTorch extension: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Check if PyTorch is installed
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} detected")
    except ImportError:
        print("⚠ Warning: PyTorch not found. Installing PyTorch...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

    # Build the Rust extension
    if "develop" in sys.argv or "install" in sys.argv:
        build_rust_extension()

    setup(
        name="rusttorch",
        version="0.1.0",
        description="High-Performance PyTorch Extension in Rust",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="RustTorch Contributors",
        url="https://github.com/yourusername/rusttorch",
        license="BSD-3-Clause",
        python_requires=">=3.10",
        install_requires=[
            "torch>=2.0.0",
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Rust",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
