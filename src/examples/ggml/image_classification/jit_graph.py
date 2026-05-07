import torch
from torch import nn
from torchvision import models
import json
import re


def get_tensor_shape(val):
    try:
        if val.type():
            type_str = str(val.type())
            match = re.search(r"Float\(([^)]+)\)", type_str)
            if match:
                dims = match.group(1).split(",")
                shape = []
                for d in dims:
                    d = d.strip()
                    if d.isdigit():
                        shape.append(int(d))
                    elif d == "-1":
                        shape.append(-1)
                return shape
        return None
    except:
        return None


def get_layer_name(node):
    try:
        scope = node.scope()
        if scope:
            return scope.replace("/", ".")
        return None
    except:
        return None


def is_conv_node(node):
    kind = node.kind()
    return kind in ["aten::_conv_2d", "aten::conv2d", "aten::conv2d.functional"]


def is_linear_node(node):
    kind = node.kind()
    return kind in ["aten::linear", "aten::linear.functional"]


def is_bn_node(node):
    kind = node.kind()
    return "batch_norm" in kind.lower()


def is_relu_node(node):
    kind = node.kind()
    return kind in [
        "aten::relu",
        "aten::relu_",
        "aten::gelu",
        "aten::silu",
        "aten::sigmoid",
        "aten::tanh",
    ]


def is_pool_node(node):
    kind = node.kind()
    return "pool" in kind.lower()


def get_layer_type(node):
    kind = node.kind()
    if is_conv_node(node):
        return "Conv2d"
    elif is_linear_node(node):
        return "Linear"
    elif is_bn_node(node):
        return "BatchNorm2d"
    elif is_relu_node(node):
        return "ReLU"
    elif is_pool_node(node):
        return "Pool"
    return None


def extract_attributes(node, layer_type):
    attrs = {}
    try:
        if layer_type == "Conv2d":
            for inp in node.inputs():
                inp_name = inp.debugName() if inp.debugName() else str(inp)
                if "stride" in inp_name.lower():
                    shape = get_tensor_shape(inp)
                    if shape:
                        attrs["stride"] = shape
                elif "padding" in inp_name.lower():
                    shape = get_tensor_shape(inp)
                    if shape:
                        attrs["padding"] = shape
                elif "kernel" in inp_name.lower() or "kernel_size" in inp_name.lower():
                    shape = get_tensor_shape(inp)
                    if shape:
                        attrs["kernel_size"] = shape
                elif "dilation" in inp_name.lower():
                    shape = get_tensor_shape(inp)
                    if shape:
                        attrs["dilation"] = shape
                elif "groups" in inp_name.lower() or "group" in inp_name.lower():
                    try:
                        attrs["groups"] = int(str(inp))
                    except:
                        pass
        elif layer_type == "Linear":
            for inp in node.inputs():
                inp_name = inp.debugName() if inp.debugName() else str(inp)
                if "bias" in inp_name.lower():
                    continue
            for out in node.outputs():
                out_type = str(out.type()) if out.type() else ""
                if "out_features" in out_type:
                    match = re.search(r"out_features=(\d+)", out_type)
                    if match:
                        attrs["out_features"] = int(match.group(1))
                if "in_features" in out_type:
                    match = re.search(r"in_features=(\d+)", out_type)
                    if match:
                        attrs["in_features"] = int(match.group(1))
    except Exception as e:
        pass
    return attrs


def get_module_name(module, prefix=""):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return f"{prefix}{name}" if prefix else name
        elif isinstance(child, nn.Linear):
            return f"{prefix}{name}" if prefix else name
        elif isinstance(child, nn.BatchNorm2d):
            return f"{prefix}{name}" if prefix else name
        elif isinstance(child, nn.ReLU):
            return f"{prefix}{name}" if prefix else name
        elif isinstance(child, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            return f"{prefix}{name}" if prefix else name
        else:
            result = get_module_name(child, f"{prefix}{name}.")
            if result:
                return result
    return None


def extract_layers_from_model(model):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append(
                {
                    "name": name,
                    "type": "Conv2d",
                    "module": module,
                    "attributes": {
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": list(module.kernel_size),
                        "stride": list(module.stride),
                        "padding": list(module.padding),
                        "dilation": list(module.dilation),
                        "groups": module.groups,
                    },
                }
            )
        elif isinstance(module, nn.Linear):
            layers.append(
                {
                    "name": name,
                    "type": "Linear",
                    "module": module,
                    "attributes": {
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                    },
                }
            )
        elif isinstance(module, nn.BatchNorm2d):
            layers.append(
                {
                    "name": name,
                    "type": "BatchNorm2d",
                    "module": module,
                    "attributes": {
                        "num_features": module.num_features,
                        "eps": module.eps,
                        "momentum": module.momentum,
                    },
                }
            )
        elif isinstance(module, nn.ReLU):
            layers.append({"name": name, "type": "ReLU", "module": module})
    return layers


def build_connections_from_graph(layers, graph):
    connections = []
    layer_outputs = {}
    layer_inputs = {}

    for layer in layers:
        layer_outputs[layer["name"]] = []
        layer_inputs[layer["name"]] = []

    for node in graph.nodes():
        node_kind = node.kind()
        layer_name = get_layer_name(node)

        if not layer_name:
            continue

        layer_found = None
        for layer in layers:
            if layer["name"] in layer_name or layer_name in layer["name"]:
                layer_found = layer
                break

        if not layer_found:
            continue

        for out in node.outputs():
            out_shape = get_tensor_shape(out)
            if out_shape:
                layer_outputs[layer_found["name"]].append({"shape": out_shape})

        for inp in node.inputs():
            inp_shape = get_tensor_shape(inp)
            if inp_shape:
                layer_inputs[layer_found["name"]].append({"shape": inp_shape})

    for i, layer in enumerate(layers):
        if layer["type"] in ["ReLU", "BatchNorm2d"]:
            for j, prev_layer in enumerate(layers[:i]):
                if prev_layer["type"] == "Conv2d" and layer["type"] in [
                    "ReLU",
                    "BatchNorm2d",
                ]:
                    if layer["name"].startswith(prev_layer["name"].split(".")[0]):
                        connections.append(
                            {"from": prev_layer["name"], "to": layer["name"]}
                        )
                        break

    return connections, layer_inputs, layer_outputs


def extract_layers_from_graph(graph):
    layers = []
    seen_layers = set()

    for node in graph.nodes():
        layer_type = get_layer_type(node)
        if not layer_type:
            continue

        layer_name = get_layer_name(node)
        if not layer_name:
            continue

        if layer_name in seen_layers:
            continue
        seen_layers.add(layer_name)

        inputs = []
        outputs = []
        for inp in node.inputs():
            shape = get_tensor_shape(inp)
            if shape:
                inputs.append(
                    {
                        "name": inp.debugName() if inp.debugName() else "input",
                        "shape": shape,
                    }
                )

        for out in node.outputs():
            shape = get_tensor_shape(out)
            if shape:
                outputs.append(
                    {
                        "name": out.debugName() if out.debugName() else "output",
                        "shape": shape,
                    }
                )

        attributes = extract_attributes(node, layer_type)

        layers.append(
            {
                "name": layer_name,
                "type": layer_type,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": attributes,
            }
        )

    return layers


def build_connections(layers):
    connections = []
    for i in range(len(layers) - 1):
        curr = layers[i]
        next_layer = layers[i + 1]
        if curr["outputs"] and next_layer["inputs"]:
            if curr["outputs"][0]["shape"] == next_layer["inputs"][0]["shape"]:
                connections.append({"from": curr["name"], "to": next_layer["name"]})
    return connections


model = models.resnet50(pretrained=True)
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
graph = traced.graph
print(type(graph))
exit()

layers_from_model = extract_layers_from_model(model)
layers_from_graph = extract_layers_from_graph(graph)

print(layers_from_graph)

result = {"layers": [], "connections": []}

for layer in layers_from_model:
    layer_data = {
        "name": layer["name"],
        "type": layer["type"],
        "inputs": [{"name": "input", "shape": "inferred"}],
        "outputs": [{"name": "output", "shape": "inferred"}],
    }
    if "attributes" in layer:
        layer_data["attributes"] = layer["attributes"]
    result["layers"].append(layer_data)

with open("resnet50_layers.json", "w") as f:
    json.dump(result, f, indent=2)
print("Saved to resnet50_layers.json")
