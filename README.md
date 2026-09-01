# ⚡ EcoEdge AI: Dynamic Neural Network Optimization Framework

**EcoEdge AI** is a lightweight, end-to-end framework designed to optimize deep learning models for deployment on resource-constrained **Edge devices**.

By combining **structural pruning**, **INT8 quantization**, and **knowledge distillation**, EcoEdge AI drastically reduces **memory footprint**, **latency**, and **power consumption** without sacrificing inference accuracy.

---

## 🚀 Key Features

### Dynamic Pruning Engine

Automatically removes redundant weights using **structured magnitude-based pruning**.

### Quantization Pipeline

Converts **FP32 models to INT8 / Bfloat16**, optimizing execution for microcontrollers and low-power CPUs.

### Automated Benchmarking

Measures:

- Real-time latency (ms)
- Peak RAM usage (MB)
- Compression ratio
- Accuracy trade-offs

### Modular & Extensible

Clean Python architecture easily adaptable to:

- Vision models
- Audio models
- IoT tabular models

---

## 🚀 Quick Start

```bash
pip install ecoedge-ai
```
---

## Usage Example

```python
import torch
import torchvision.models as models

from ecoedge.compression import Pruner
from ecoedge.export import ONNXExporter

# 1. Load your PyTorch model
model = models.resnet18(pretrained=True)

# 2. Prune 30% of Conv2d channels
pruner = Pruner(model)

pruned_model = pruner.prune_l1_structured(amount=0.3)

# 3. Export to ONNX for Edge deployment
exporter = ONNXExporter(pruned_model)

exporter.export("resnet18_edge.onnx")
```
