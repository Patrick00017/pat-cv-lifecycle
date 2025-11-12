import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import Optional, List, Dict
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from torchvision.models.resnet import resnet50

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # check if finetune the backbone
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}    
        else:
            return_layers = {'layer4': '0'}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # expand to every layer
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        backbone = resnet50(weights='IMAGENET1K_V1')
        num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos