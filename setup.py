# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "fastapi==0.111.0",
    "pydantic>=2.7,<3.0",
    "requests==2.31.0",
    "rich==13.4.2",
    "redis==5.0.4",
    "fakeredis==2.30.0",
    "neo4j==5.17.0",
    "boto3==1.34.0",
    "gudhi==3.9.0",
    "numpy==1.26.0",
    "pillow==11.0.0",
    "prometheus-client==0.22.0",
    "sqlalchemy==2.0.45",
    "scipy==1.13.1",
    "networkx==3.2.1",
    # "faiss-cpu==1.8.0", # Conflicts with faiss-gpu often; trying to install all might mean preferring GPU if listed?
    # User said "all dependencies", I will list both but pip might complain or overwrite.
    "faiss-cpu==1.8.0",
    # "faiss-gpu>=1.8.0", # Removed: caused installation failure on Windows
    "torch==2.3.1",
    "torchvision==0.18.1",
    "transformers==4.44.2",
    "bitsandbytes==0.43.3",
    "scikit-learn==1.4.2",
    "imagehash==4.3.1",
    "pybreaker==1.0.0",
    "python-docx==1.1.0",
    "python-pptx==0.6.23",
    "pdfminer.six==20231228",
    "pytesseract==0.3.10",
    "unstructured==0.15.0",
    "langdetect==1.0.9",
    "pytubefix==8.1.1",
    "youtube-transcript-api==0.6.2",
    "pyarrow==16.1.0",
    "scikit-optimize==0.10.2",
    "watchdog==4.0.0",
    "tomlkit>=0.13.0",
    # Merged from EXTRAS
    "webrtcvad==2.0.10",
    # "cupy-cuda12x", # Removed: likely to fail if no CUDA
    "ray[serve]==2.52.1",
]

setup(
    name="datacreek",
    version="0.0.0",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require={}, # Cleared as requested to force install all
)
