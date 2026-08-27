import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns

from ecoedge.core.utils import get_device
from ecoedge.compression.pruner import StructuredPruner
from ecoedge.compression.quantizer import ModelQuantizer
from ecoedge.profiling.latency_meter import LatencyMeter
from ecoedge.profiling.memory_tracker import MemoryTracker
from ecoedge.export.onnx_exporter import ONNXExporter


def main():
    device = get_device()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    baseline_model = torchvision.models.resnet18(num_classes=10).to(device)
    baseline_model.eval()

    base_size = MemoryTracker.get_model_size_mb(baseline_model)
    base_lat, _ = LatencyMeter.measure_inference_time(
        baseline_model, dummy_input
    )

    pruner = StructuredPruner(baseline_model)
    prune_model = pruner.apply_structured_pruning(amount=0.3)
    prune_lat, _ = LatencyMeter.measure_inference_time(
        prune_model, dummy_input
    )

    quantizer = ModelQuantizer(baseline_model.cpu())
    quant_model = quantizer.quantize_dynamic_int8()
    quant_size = MemoryTracker.get_model_size_mb(quant_model)
    quant_lat, _ = LatencyMeter.measure_inference_time(
        quant_model, dummy_input.cpu()
    )

    os.makedirs("exports", exist_ok=True)
    onnx_path = "exports/ecoedge_model.onnx"
    ONNXExporter.export_to_onnx(baseline_model.cpu(), dummy_input.cpu(), onnx_path)

    print("\n=== Benchmark Results ===")
    print(f"Baseline: Size = {base_size:.2f} MB | Latency = {base_lat:.2f} ms")
    print(f"Quantized INT8: Size = {quant_size:.2f} MB | Latency = {quant_lat:.2f} ms")
    print(f"Pruned Model Latency: {prune_lat:.2f} ms")

    categories = ["Baseline", "Optimized"]
    latencies = [base_lat, quant_lat]
    sizes = [base_size, quant_size]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.barplot(x=categories, y=latencies, ax=axes[0], palette="viridis")
    axes[0].set_title("Inference Latency (ms)")
    axes[0].set_ylabel("ms")

    sns.barplot(x=categories, y=sizes, ax=axes[1], palette="magma")
    axes[1].set_title("Model Size (MB)")
    axes[1].set_ylabel("MB")

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("Benchmark chart saved as benchmark_results.png")


if __name__ == "__main__":
    main()
