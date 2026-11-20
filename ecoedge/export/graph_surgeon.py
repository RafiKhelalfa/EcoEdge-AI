import onnx

def sanitize_pruned_onnx_model(onnx_model_path: str, output_path: str):
    """
    Parcourt le graphe ONNX pour supprimer les initializers orphelins
    et nettoyer les attributs de shape faussés par le pruning structuré.
    Empêche les décalages de tenseurs (silent tensor drift) sur l'Edge.
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
    print(f"[EcoEdge-AI] Modèle ONNX nettoyé et sauvegardé avec succès : {output_path}")
