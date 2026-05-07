# import torch
# from torch import nn
# from torchvision import models
# import json

# model = models.resnet50(pretrained=True)
# model.eval()

# script_model = torch.jit.script(model)
# print(script_model)
from torchvision import models
from typing import List
import torch


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(">>> my_compiler() invoked:")
    print(">>> FX graph:")
    gm.graph.print_tabular()
    # print(f">>> Code:\n{gm.code}")
    return gm.forward  # return a python callable


model = models.resnet50(pretrained=True)


@torch.compile(backend=my_compiler, fullgraph=True)
def foo(x):
    return model(x)


if __name__ == "__main__":
    # a, b = torch.randn(10), torch.ones(10)
    x = torch.randn((1, 3, 224, 224))
    foo(x)
