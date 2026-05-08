# Dog cat classification for GGML

## TODO

- [ ] torch JIT export compute graph 

## Answer from MiniMax

How torch.onnx.export Gets the Compute Graph
torch.onnx.export obtains the compute graph through tracing (or scripting as an alternative):
1. Tracing (Default for nn.Module)
When the model is a regular torch.nn.Module, torch.onnx.export runs the model once with example inputs using torch.jit.trace():
- Executes the model with the provided args
- Records all tensor operations during execution
- Captures a static compute graph representing only that run
Key limitation: Dynamic control flow (if/else based on data, loops) gets unrolled into a static graph.
2. Scripting (For ScriptModule)
If the model is already a torch.jit.ScriptModule or ScriptFunction, it uses the existing TorchScript graph directly via model.forward.graph.
3. Modern Approach (PyTorch 2.5+)
With dynamo=True (now the default), torch.onnx.export uses torch.export which:
- Uses TorchDynamo bytecode analysis to trace the computation
- Produces an ExportedProgram with an FX graph in "ATen IR" (low-level operators)
- Supports non-strict mode (strict=False) for handling more Python features
Workflow Summary:
torch.nn.Module + args → torch.jit.trace() → TorchScript Graph → ONNX Export → ONNX Model
The tracing process internally calls torch.jit._get_trace_graph() which captures operations as nodes, then converts each node to its ONNX equivalent using registered symbolic functions.

## Blog 

1. https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html
2. https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html