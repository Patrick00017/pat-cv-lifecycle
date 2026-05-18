import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from torch.export.graph_signature import InputSpec, OutputSpec
from torchvision import datasets, transforms, models


class AutoGenerateUtils:
    def __init__(self, model: nn.Module, input_args):

        self.exported_program: ExportedProgram = export(model, args=input_args)
        self.graph_signature = self.exported_program.graph_signature
        self.graph = self.exported_program.graph
        self.input_specs: list[InputSpec] = self.graph_signature.input_specs
        self.output_specs: list[OutputSpec] = self.graph_signature.output_specs

    def check_input(self):
        for i in range(5):
            print(self.input_specs[i].kind)  # parameter or buffer, the same in ggml?
            print(self.input_specs[i].arg)  # .name is the name in exported graph
            print(self.input_specs[i].target)  # weight name in state_dict
            # persistent=True means the buffer is saved to and loaded from the model's state_dict() when saving/loading checkpoints. persistent=False buffers (like num_batches_tracked) are not saved - they are recalculated at runtime.
            print(self.input_specs[i].persistent)

    def check_graph(self):
        # print(self.graph)
        print(self.graph.__str__)
        for node in self.graph.nodes:
            # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
            # print(node.op)  # call_function
            # print(node.name)  # node name
            # print(node.target)  # function name to be called, like aten.conv2d.default
            # print(node.type)  # None
            # print(
            # node.all_input_nodes
            # )  # a dict like {relu__47: None, p_layer4_2_conv3_weight: None}
            # print(node.kwargs)
            print(f"""
Node:
    name: {node.name},
    op: {node.op},
    target: {node.target},
    type: {node.type},
    input_nodes: {node.all_input_nodes},
    args: {node.args}
    kwargs: {node.kwargs},
    meta: {node.meta}
""")

    def generate(self):
        pass


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    input_args = (torch.randn(1, 3, 256, 256),)
    tool = AutoGenerateUtils(model=model, input_args=input_args)
    # tool.check_input()
    tool.check_graph()
