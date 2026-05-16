import torch
from torch.export import export, ExportedProgram


# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))


example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print("exported_program: ")
print(exported_program)
# torch.export.save(exported_program, "exported_program.pt2")
print("graph: ")
print(exported_program.graph)
print("graph module: ")
print(exported_program.graph_module)
# To run the exported program, we can use the `module()` method
# print(
#     exported_program.module()(
#         torch.randn(1, 3, 256, 256), constant=torch.ones(1, 16, 256, 256)
#     )
# )
