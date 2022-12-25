import types

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, MarianMTModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from adaptor.evaluators.generative import GenerativeEvaluator, BLEU, BERTScore, ROUGE
from adaptor.utils import AdaptationDataset


class EnsembleEvaluator(GenerativeEvaluator):

    def __init__(self, *args, original_model: PreTrainedModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model = original_model

    @staticmethod
    def _get_ensemble_model(orig_model: PreTrainedModel, trained_model: PreTrainedModel) -> PreTrainedModel:

        # class EnsembleModel(torch.nn.Module):
        #     ensemble_model = EnsembleModel()  # avoid calling __init__, that would redundantly allocate new model params
        #     ensemble_model.config = trained_model.config
        #     ensemble_model.model = trained_model.model
        #     ensemble_model.lm_head = trained_model.lm_head
        #     ensemble_model.device = trained_model.device
        #     # TODO: continue this way: AttributeError: 'MarianMTModel' object has no attribute 'lm_head'
        #     ensemble_model.__class__ = trained_model.__class__

        def forward(self, *args, **kwargs):
            orig_output = MarianMTModel.forward(orig_model, **{k: v for k, v in kwargs.items() if v is not None})
            trained_output = MarianMTModel.forward(trained_model, **{k: v for k, v in kwargs.items() if v is not None})
            # assuming Seq2SeqLMOutput as forwards' return types
            # see transformers.marian.modeling_marian.MarianMTModel
            ensemble_output = Seq2SeqLMOutput(loss=trained_output,
                                              logits=(trained_output.logits + orig_output.logits),
                                              past_key_values=trained_output.past_key_values,
                                              decoder_hidden_states=trained_output.decoder_hidden_states,
                                              decoder_attentions=trained_output.decoder_attentions,
                                              cross_attentions=trained_output.cross_attentions,
                                              encoder_last_hidden_state=trained_output.encoder_last_hidden_state,
                                              encoder_hidden_states=trained_output.encoder_hidden_states,
                                              encoder_attentions=trained_output.encoder_attentions)
            return ensemble_output

        # setattr(trained_model, "orig_forward", types.MethodType(trained_model.forward, trained_model))
        setattr(trained_model, "forward", types.MethodType(forward, trained_model))

        return trained_model

    def __call__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, dataset: AdaptationDataset) -> float:
        ensemble_model = self._get_ensemble_model(self.original_model, model)
        return super(EnsembleEvaluator, self).__call__(ensemble_model, tokenizer, dataset)


class EnsembleBLEU(EnsembleEvaluator, BLEU):
    pass


class EnsembleROUGE(EnsembleEvaluator, ROUGE):
    pass


class EnsembleBERTScore(EnsembleEvaluator, BERTScore):
    pass
