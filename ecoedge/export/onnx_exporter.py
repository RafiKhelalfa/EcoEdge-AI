import torch
import torch.nn as nn
import onnx
import onnxruntime as ort


class ONNXExporter:

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        export_path: str,
        opset_version: int = 13,
    ) -> None:
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)

    @staticmethod
    def create_ort_session(model_path: str) -> ort.InferenceSession:
        return ort.InferenceSession(model_path)
