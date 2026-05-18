import torch
import torch.nn as nn
from torch.export import export, ExportedProgram
from torch.export.graph_signature import InputSpec, OutputSpec


class AutoGenerate:
    def __init__(self, model: nn.Module, input_args):
        # input_args = (torch.randn(1, 3, 256, 256),)
        self.exported_program: ExportedProgram = export(model, args=input_args)
        self.graph_signature = self.exported_program.graph_signature
        self.graph = self.exported_program.graph
        self.input_specs: list[InputSpec] = self.graph_signature.input_specs
        self.output_specs: list[OutputSpec] = self.graph_signature.output_specs
