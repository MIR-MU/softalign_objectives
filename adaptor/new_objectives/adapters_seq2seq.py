import logging
from typing import Dict, Any

import torch
from transformers import AutoModelForSeq2SeqLM, MarianMTModel, AutoConfig

from ..objectives.objective_base import SupervisedObjective
from ..objectives.seq2seq import Sequence2SequenceMixin
from ..utils import Head

logger = logging.getLogger()


class Adapter(torch.nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """

    def __init__(self, in_dim=512, reduction_factor=32, weights_init_std: float = 10e-2):
        super().__init__()
        self.adapter_block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim // reduction_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim // reduction_factor, in_dim),
        )
        for layer in self.adapter_block:
            with torch.no_grad():
                if hasattr(layer, "weight"):  # applies to i \in [0, 2]
                    torch.nn.init.normal_(layer.weight, mean=0, std=weights_init_std)
                    # prune to 2x STD (Houlsby et al, 2019)
                    layer.weight[layer.weight < -weights_init_std*2] = -weights_init_std*2
                    layer.weight[layer.weight > weights_init_std*2] = weights_init_std*2

    def forward(self, x):
        ff_out = self.adapter_block(x)
        # Skip connection
        adapter_out = ff_out + x

        return adapter_out


class AdapterWrapper(torch.nn.Module):
    def __init__(self, orig_layer):
        super().__init__()
        self.orig_layer = orig_layer
        # self.orig_layer.training = False
        self.adapter = Adapter(in_dim=self.orig_layer.out_features)

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        adapter_output = self.adapter(orig_out)

        return adapter_output


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
            enc_layer.self_attn.out_proj = AdapterWrapper(enc_layer.self_attn.out_proj)
            enc_layer.fc2 = AdapterWrapper(enc_layer.fc2)

        for dec_layer in self.model.decoder.layers:
            dec_layer.self_attn.out_proj = AdapterWrapper(dec_layer.self_attn.out_proj)
            dec_layer.fc2 = AdapterWrapper(dec_layer.fc2)

        logger.warning("Initialized Adapter layers")


class AdapterSequence2Sequence(Sequence2SequenceMixin, SupervisedObjective):

    compatible_head = Head.SEQ2SEQ_ADAPTER
