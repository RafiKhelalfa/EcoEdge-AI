from .pruner import StructuredPruner
from .quantizer import ModelQuantizer
from .distiller import KnowledgeDistiller
import torch
import torchvision.models as models

# Importer directement depuis le fichier pruning.py
from ecoedge.compression.pruning import Pruner
from ecoedge.export.onnx_exporter import ONNXExporter

__all__ = ["StructuredPruner", "ModelQuantizer", "KnowledgeDistiller"]
