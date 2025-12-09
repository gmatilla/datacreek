from setuptools import find_packages, setup

setup(
    name="datacreek",
    version="0.0.0",
    packages=find_packages(),
    extras_require={
        "gpu": ["cupy-cuda12x", "faiss-gpu>=1.8.0"],
    },
)
