import abc
import types
from typing import Dict, Any, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, MarianMTModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from adaptor.evaluators.generative import GenerativeEvaluator, BLEU, BERTScore, ROUGE
from adaptor.utils import AdaptationDataset


class EnsembleEvaluator(GenerativeEvaluator, abc.ABC):

    def __init__(self, *args, original_model: PreTrainedModel, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_model = original_model

    @staticmethod
    def _get_ensemble_model(orig_model: PreTrainedModel, trained_model: PreTrainedModel) -> PreTrainedModel:
        def forward(self, *args, **kwargs):
            # naming convention to pass hidden states of correct encoder
            orig_args = {k: v for k, v in kwargs.items() if v is not None and not k.startswith("encoder")}
            orig_args = {k.replace("orig_", ""): v for k, v in orig_args.items()}

            trained_args = {k: v for k, v in kwargs.items() if v is not None and not k.startswith("orig")}
            if "encoder_outputs" in trained_args and "encoder_outputs" in orig_args:
                assert id(trained_args["encoder_outputs"]) != id(orig_args["encoder_outputs"])

            orig_output = MarianMTModel.forward(orig_model, **orig_args)
            trained_output = MarianMTModel.forward(trained_model, **trained_args)
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

        def _prepare_encoder_decoder_kwargs_for_generation(self,
                                                           input_ids: torch.LongTensor,
                                                           model_kwargs) -> Dict[str, Any]:
            if "encoder_outputs" not in model_kwargs:
                # retrieve encoder hidden states
                encoder_kwargs = {argument: value for argument, value in model_kwargs.items()
                                  if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))}

                encoder = self.get_encoder()
                model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids,
                                                                       return_dict=True,
                                                                       **encoder_kwargs)

                orig_encoder = orig_model.get_encoder()
                model_kwargs["orig_encoder_outputs"]: ModelOutput = orig_encoder(input_ids,
                                                                                 return_dict=True,
                                                                                 **encoder_kwargs)
            return model_kwargs

        def prepare_inputs_for_generation(self, *args, **kwargs):
            inputs = MarianMTModel.prepare_inputs_for_generation(self, *args, **kwargs)

            inputs["orig_encoder_outputs"] = kwargs["orig_encoder_outputs"]
            return inputs

        def _expand_inputs_for_generation(self,
                                          input_ids: torch.LongTensor,
                                          expand_size: int = 1,
                                          is_encoder_decoder: bool = False,
                                          attention_mask: torch.LongTensor = None,
                                          encoder_outputs: ModelOutput = None,
                                          **model_kwargs) -> Tuple[torch.LongTensor, Dict[str, Any]]:
            expanded_return_idx = (
                torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
            )
            input_ids = input_ids.index_select(0, expanded_return_idx)

            if "token_type_ids" in model_kwargs:
                token_type_ids = model_kwargs["token_type_ids"]
                model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

            if is_encoder_decoder:
                assert encoder_outputs is not None
                encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
                )
                orig_encoder_outputs = model_kwargs["orig_encoder_outputs"]
                orig_encoder_outputs["last_hidden_state"] = orig_encoder_outputs.last_hidden_state.index_select(
                    0, expanded_return_idx.to(orig_encoder_outputs.last_hidden_state.device)
                )
                model_kwargs["encoder_outputs"] = encoder_outputs
                model_kwargs["orig_encoder_outputs"] = orig_encoder_outputs
            return input_ids, model_kwargs

        # setattr(trained_model, "orig_forward", types.MethodType(trained_model.forward, trained_model))
        setattr(trained_model, "forward", types.MethodType(forward, trained_model))
        setattr(trained_model, "_prepare_encoder_decoder_kwargs_for_generation",
                types.MethodType(_prepare_encoder_decoder_kwargs_for_generation, trained_model))
        setattr(trained_model, "prepare_inputs_for_generation",
                types.MethodType(prepare_inputs_for_generation, trained_model))
        setattr(trained_model, "_expand_inputs_for_generation",
                types.MethodType(_expand_inputs_for_generation, trained_model))

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
