import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from einops import rearrange, repeat
from typing import Optional, List, Dict
from scipy.optimize import linear_sum_assignment
from modules.backbone import Backbone, Joiner
from utils.box import generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from modules.position_encoding import PositionEmbeddingSine
from modules.transformer import Transformer


# show tensor shape in vscode debugger
def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(pl.LightningModule):
    def __init__(self, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = Transformer()
        resnet50 = Backbone("resnet50", False, True, False)
        pos_embed = PositionEmbeddingSine(num_pos_feats=256)
        backbone_with_pos_embed = Joiner(resnet50, pos_embed)
        self.backbone = backbone_with_pos_embed
        transformer = Transformer()
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        self.hungarion_matcher = HungarianMatcher()

    def forward(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )[0]
        # hs (1, num_queries, bs, dim)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.forward(images)
        matches = self.hungarion_matcher(output, targets)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        # optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, weight_decay=0.0005)
        optimizer = torch.optim.SGD(
            [p for p in self.parameters() if p.requires_grad],
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005,
        )
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return {"optimizer": optimizer}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class HungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # (bs*num_queries, num_classes)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (bs*num_queries, 4)

        # concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  # (bs*n, 1)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # (bs*n, 4)

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(
            bs, num_queries, -1
        ).cpu()  # C.shape = (bs, num_queries, total_targets_across_all_batches)

        sizes = [len(v["boxes"]) for v in targets]
        # here i -> bs, c -> bboxes, perfect trick.
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


if __name__ == "__main__":
    backbone = Backbone("resnet50", False, True, False)
    pos_embed = PositionEmbeddingSine(num_pos_feats=256)
    joiner = Joiner(backbone, pos_embed)
    transformer = Transformer()
    detr = DETR(joiner, transformer, 10, 100)
    x = [torch.randn((3, 400, 600)), torch.randn((3, 600, 400))]
    # y = []
    output = detr(x)
    pred_logits = output["pred_logits"]
    pred_boxes = output["pred_boxes"]
    # print(output)
    print(f"pred_logits: {pred_logits.shape}")  # (2,100,11)
    print(f"pred_boxes: {pred_boxes.shape}")  # (2, 100, 4)
