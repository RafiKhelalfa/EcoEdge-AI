⚡ EcoEdge AI: Dynamic Neural Network Optimization Framework
===========================================================

[![PyPI version](https://badge.fury.io/py/ecoedge-ai.svg)](https://badge.fury.io/py/ecoedge-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xGCFo07ijkGJ-fy1zJheygDQzNnPRh2R)

**EcoEdge AI** is a lightweight, end-to-end framework designed to optimize deep learning models for deployment on resource-constrained **Edge devices**.

By combining **structural pruning**, **INT8 quantization**, and **knowledge distillation**, EcoEdge AI drastically reduces **memory footprint**, **latency**, and **power consumption** without sacrificing inference accuracy.

## 🚀 Quick Start
--------------

### 1\. Installation
```
pip install ecoedge-ai
```

### 2\. Basic Usage (Scikit-Learn Style)

```
import torch

import torchvision.models as models

from ecoedge.compression import Pruner

from ecoedge.export import ONNXExporter

# Load your PyTorch model

model = models.resnet18(pretrained=True)

# Prune 30% of Conv2d channels

pruner = Pruner(model)

pruned_model = pruner.prune_l1_structured(amount=0.3)

# Export to ONNX for Edge deployment

exporter = ONNXExporter(pruned_model)

exporter.export("resnet18_edge.onnx")
```

## 📊 Performance & Benchmark Results
----------------------------------

EcoEdge AI drastically reduces edge model footprints and inference times using dynamic INT8 quantization and structured pruning.

### Key Metrics Summary

| **Metric** | **Baseline (FP32)** | **EcoEdge AI (Optimized)** | **Improvement** |
| --- | --- | --- | --- |
| **Inference Latency** | 18.10 ms | **6.70 ms** | **+63% Speedup** |
| **Model File Size** | 42.69 MB | **15.79 MB** | **-63% Memory** |
| **Execution Target** | CPU Baseline | CPU Edge Optimized | High Efficiency |

## 📜 License
----------

Distributed under the **MIT License**.

See [`LICENSE`](https://github.com/RafiKhelalfa/EcoEdge-AI/blob/main/LICENSE) for more information.
