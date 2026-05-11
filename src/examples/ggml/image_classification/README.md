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

## Understand how torch generate the code based on the graph

How PyTorch FX Generates Code from a Graph
The code generation pipeline involves two main files:
1. graph_module.py - Orchestration Layer
Key flow: GraphModule → recompile() → Graph.python_code()
# graph_module.py - recompile() method (line 590)
def recompile(self) -> PythonCode:
    python_code = self._graph.python_code(       # <-- delegates to Graph
        root_module="self",
        record_func=fx_experimental_config.enrich_profiler_metadata,
    )
    self._code = python_code.src                  # <-- extracts generated source
    cls.forward = _forward_from_src(self._code, ...)  # <-- compiles to method
2. graph.py - The Actual Code Generator
The real logic lives in torch/fx/graph.py, specifically in the python_code() method. The core codegen classes are:
- _PyTreeCodeGen — Handles input/output pytree (nested dict/list) flattening
- _BoxedCodeGen — Wraps forward in a box for deferred calling
- _Custom成都CodeGen — Handles user-defined code generation
The Core Algorithm (from graph.py)
1. Emit prologue — Generates function signature from in_spec (input structure)
2. Emit nodes — Iterates graph nodes in topological order, generating Python statements for each op type:
   - placeholder → function parameter
   - get_attr → self.xxx
   - call_function → direct function call
   - call_module → self.submodule(...)
   - call_method → obj.method(...)
   - output → return statement
3. Emit epilogue — Wraps outputs back into pytree structure using out_spec
Key Codegen Hooks
Hook	Purpose
add_pixel_value(...)	Override how tensor constants are emitted
add_extra_imports(...)	Add custom imports
_custom_builtins	Built-in functions/classes injected into generated code