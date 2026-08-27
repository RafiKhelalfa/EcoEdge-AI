from setuptools import setup, find_packages

setup(
    name="ecoedge",
    version="0.1.0",
    author="EcoEdge AI Team",
    description="Dynamic AI model optimization framework for edge devices.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "onnx",
        "onnxruntime",
        "matplotlib",
        "seaborn",
        "psutil",
        "pyyaml",
        "numpy",
    ],
)
