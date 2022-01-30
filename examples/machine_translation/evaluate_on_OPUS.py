"""
Evaluate adapted checkpoint on selected OPUS data sets.
"""
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from adaptor.evaluators.generative import BLEU
from examples.data_utils_opus import OPUSDataset, OPUS_RESOURCES_URLS

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', type=str, help='Path of the fine-tuned checkpoint.',
                           required=True)
    argparser.add_argument('--opus_dataset', type=str, required=True,
                           help='One of the recognized OPUS datasets: %s' % OPUS_RESOURCES_URLS)
    argparser.add_argument('--no_generate', default=False, action='store_true',
                           help='Whether to use model.generate() method for test predictions.')
    argparser.add_argument('--src_lang', type=str, required=True, help='Source language of the OPUS data set.')
    argparser.add_argument('--tgt_lang', type=str, required=True, help='Target language of the OPUS data set.')
    argparser.add_argument('--data_dir', type=str, required=True, help='Cache directory to store the data.')
    argparser.add_argument('--firstn', type=int, default=None, help='If given, subsets data set to first-n samples.')
    argparser.add_argument('--device', type=str, default="cpu", help='Device for inference. Defaults to CPU.')
    args = argparser.parse_args()

    test_dataset = OPUSDataset(args.opus_dataset, src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                               split="test", data_dir=args.data_dir, firstn=args.firstn)

    lm_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    assert hasattr(lm_model, "generate"), "For translation, we need a model that implements its own generate()."

    metric = BLEU(use_generate=True, additional_sep_char="▁", progress_bar=False)

    references = []
    hypotheses = []
    for src_text, ref_text in zip(test_dataset.source, test_dataset.target):
        references.append(ref_text)
        inputs = tokenizer(src_text, truncation=True, return_tensors="pt").to(args.device)

        outputs = lm_model.generate(**inputs)
        translations = tokenizer.batch_decode(outputs, remove_special_tokens=True)
        hypotheses.append(translations[0])

    bleu = BLEU().evaluate_str(references, hypotheses)
    print("Experiment Test %s on %s (%s->%s): %s" %
          (metric, args.opus_dataset, args.src_lang, args.tgt_lang, bleu))
