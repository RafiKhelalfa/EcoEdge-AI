# ⚡ EcoEdge AI: Dynamic Neural Network Optimization Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**EcoEdge AI** is a lightweight, end-to-end framework designed to optimize deep learning models for deployment on resource-constrained Edge devices. By combining **structural pruning**, **INT8 quantization**, and **knowledge distillation**, EcoEdge AI drastically reduces memory footprint, latency, and power consumption without sacrificing inference accuracy.

---

## 🚀 Key Features

* **Dynamic Pruning Engine:** Automatically removes redundant weights using structured magnitude-based pruning.
* **Quantization Pipeline:** Converts FP32 models to INT8 / Bfloat16, optimizing execution for microcontrollers and low-power CPUs.
* **Automated Benchmarking:** Measures real-time latency (ms), peak RAM usage (MB), compression ratio, and accuracy trade-offs.
* **Modular & Extensible:** Clean Python architecture easily adaptable to vision, audio, or IoT tabular models.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/RafiKhelalfa/EcoEdge-AI.git
cd EcoEdge-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
