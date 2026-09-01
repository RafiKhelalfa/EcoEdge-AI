from setuptools import setup, find_packages

setup(
    name="ecoedge-ai",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "onnxscript",
    ],
    author="Rafi Khelalfa",
    description="A lightweight framework for Deep Learning model pruning, INT8 quantization, and ONNX export for edge devices.",
    url="https://github.com/RafiKhelalfa/EcoEdge-AI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
