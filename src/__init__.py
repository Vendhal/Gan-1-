"""
src package for Vanilla GAN project.

This file makes `src` a proper Python package so that
absolute imports like `from src.generator import Generator`
work consistently with:
- uvicorn
- streamlit
- multiprocessing (Windows)
"""

__all__ = [
    "generator",
    "discriminator",
    "inference",
    "monitoring",
    "train",
    "api",
    "app",
]


