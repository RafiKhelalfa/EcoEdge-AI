# EcoEdge-AI 🌿

EcoEdge-AI is a cutting-edge Python framework designed for efficient AI model compression, structured pruning, adaptive quantization, and ONNX export optimized for edge devices.

---

## 🚀 New Advanced Features (Branch: `feature/advanced-edge-modules`)

### 1. ONNX Graph Surgeon (`ecoedge/export/graph_surgeon.py`)
* **Problem Solved:** Prevents **Ghost Gradient Mismatch** and **silent tensor drift** on Edge devices caused by orphaned initializers and distorted shape attributes after structured pruning and dynamic axis ONNX export.
* **Usage:**
```python
from ecoedge.export.graph_surgeon import sanitize_pruned_onnx_model

sanitize_pruned_onnx_model("model.onnx", "ecoedge_optimized.onnx")
```

### 2. Adaptive Piecewise Quantizer (`ecoedge/compression/adaptive_quantizer.py`)
* **Problem Solved:** Eliminates **Zero-Point Drift** and NPU register saturation when quantizing sparse, bimodal weight distributions resulting from fine-grained pruning.
* **Usage:**
```python
from ecoedge.compression.adaptive_quantizer import AdaptivePiecewiseQuantizer

quantizer = AdaptivePiecewiseQuantizer(model, bits=8)
quantizer.apply_soft_threshold_quantization()
```

---

## 📦 Installation & PyPI Release

To build and update your package on PyPI:

```bash
pip install --upgrade build twine
rm -rf build dist *.egg-info
python -m build
python -m twine upload dist/*
```
