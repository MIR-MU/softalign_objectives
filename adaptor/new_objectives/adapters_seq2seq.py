from typing import Dict, Any

import torch
from transformers import AutoModelForSeq2SeqLM, MarianMTModel, AutoConfig

from ..objectives.objective_base import SupervisedObjective
from ..objectives.seq2seq import Sequence2SequenceMixin
from ..utils import Head


class Adapter(torch.nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """

    def __init__(self, in_dim=512, reduction_factor=2):
        super().__init__()
        self.adapter_block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // reduction_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim // reduction_factor, in_dim)
        )

    def forward(self, x):
        ff_out = self.adapter_block(x)
        # Skip connection
        adapter_out = ff_out + x

        return adapter_out


class Adaptered(torch.nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter(in_dim=self.orig_layer.out_features)

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        output = (self.adapter.forward(orig_out[0].unsqueeze(0))[0])

        return output


class AdapterTransformer(MarianMTModel):
    # inspired by https://discuss.pytorch.org/t/insert-adapters-in-a-transformer/165165/2
    # updated for a consistency with Houlsby et al., (2019): https://arxiv.org/pdf/1902.00751.pdf

    def __init__(self,
                 model_name_or_path: str,
                 **head_kwargs: Dict[str, Any]):
        super().__init__(config=AutoConfig.from_pretrained(model_name_or_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=False, **head_kwargs).model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False
        # Embed adapter layers into the transformer blocks
        for enc_layer in self.model.encoder.layers:
            enc_layer.self_attn.out_proj = Adaptered(enc_layer.self_attn.out_proj)
            enc_layer.fc2 = Adaptered(enc_layer.fc2)

        for dec_layer in self.model.decoder.layers:
            dec_layer.self_attn.out_proj = Adaptered(dec_layer.self_attn.out_proj)
            dec_layer.fc2 = Adaptered(dec_layer.fc2)


class AdapterSequence2Sequence(Sequence2SequenceMixin, SupervisedObjective):

    compatible_head = Head.SEQ2SEQ_ADAPTER
