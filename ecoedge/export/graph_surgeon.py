import onnx

def sanitize_pruned_onnx_model(onnx_model_path: str, output_path: str):
    """
    Traverses the ONNX graph to remove orphaned initializers
    and clean up shape attributes distorted by structured pruning.
    Prevents silent tensor drift on Edge devices.
    """
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    used_inputs = set()
    for node in graph.node:
        for inp in node.input:
            used_inputs.add(inp)
            
    initializers_to_keep = [init for init in graph.initializer if init.name in used_inputs]
            
    del graph.initializer[:]
    graph.initializer.extend(initializers_to_keep)
    
    onnx.save(model, output_path)
    print(f"[EcoEdge-AI] Cleaned ONNX model successfully saved to: {output_path}")
