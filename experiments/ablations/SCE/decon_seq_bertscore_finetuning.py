import comet_ml  # logging hook must be imported before torch # noqa F401
import torch

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU, ROUGE, BERTScore
from adaptor.lang_module import LangModule
from adaptor.new_objectives.seq_bertscr_objectives import SeqAlignCEDecObjective
from adaptor.objectives.seq2seq import Sequence2Sequence
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from utils.data_utils_opus import OPUSDataset

# import gc

# torch.autograd.set_detect_anomaly(True)
# gc.set_debug(gc.DEBUG_LEAK)

data_dir = "utils"
experiment_id = "mrt"

adapt_dataset = "OpenSubtitles"
test_datasets = ["wikimedia", "OpenSubtitles", "Bible"]

src_lang = "en"
tgt_lang = "cs"

# 1. Load OPUS domain-specific data sets
train_firstn = None  # no limit
val_firstn = 500
test_firstn = 1000

wiki_pairs = OPUSDataset("wikimedia", "train", src_lang, tgt_lang, data_dir=data_dir, firstn=train_firstn)
wiki_val_pairs = OPUSDataset("wikimedia", "val", src_lang, tgt_lang, data_dir=data_dir, firstn=val_firstn)

opensub_pairs = OPUSDataset("OpenSubtitles", "train", src_lang, tgt_lang, data_dir=data_dir, firstn=train_firstn)
opensub_val_pairs = OPUSDataset("OpenSubtitles", "val", src_lang, tgt_lang, data_dir=data_dir, firstn=val_firstn)

bible_pairs = OPUSDataset("Bible", "train", src_lang, tgt_lang, data_dir=data_dir, firstn=train_firstn)
bible_val_pairs = OPUSDataset("Bible", "val", src_lang, tgt_lang, data_dir=data_dir, firstn=val_firstn)

# 2. Initialize training arguments
# We apply NUM_STEPS stopping strategy in cases where at least one of the objectives does not converge in max_steps
training_arguments = AdaptationArguments(output_dir=experiment_id,
                                         learning_rate=2e-6,  # we set LR=2e-4 for pre-training experiments
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         # stopping_strategy=StoppingStrategy.NUM_STEPS_ALL_OBJECTIVES,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=100000,
                                         gradient_accumulation_steps=10,
                                         logging_steps=100,
                                         eval_steps=50,
                                         save_steps=5000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps",
                                         also_log_converged_objectives=True,
                                         stopping_patience=50)

# we initialise base model from HF model
lang_module = LangModule("Helsinki-NLP/opus-mt-en-cs")

metrics_args = {"additional_sep_char": "▁"}

val_metrics = [BLEU(**metrics_args, decides_convergence=True),
               ROUGE(**metrics_args), BERTScore(**metrics_args)]

# declaration of *all* used objectives: both training and evaluation ones (see configurations below)
tokenbsc_wiki = SeqAlignCEDecObjective(lang_module,
                                       texts_or_path=wiki_pairs.source,
                                       labels_or_path=wiki_pairs.target,
                                       val_texts_or_path=wiki_val_pairs.source[:20],
                                       val_labels_or_path=wiki_val_pairs.target[:20],
                                       source_lang_id=src_lang,
                                       target_lang_id=tgt_lang,
                                       batch_size=1,
                                       val_evaluators=val_metrics,
                                       objective_id="Wiki",
                                       loss_weight=100,
                                       remember_last_input=True)

seq_wiki = Sequence2Sequence(lang_module,
                             texts_or_path=wiki_pairs.source,
                             labels_or_path=wiki_pairs.target,
                             val_texts_or_path=wiki_val_pairs.source,
                             val_labels_or_path=wiki_val_pairs.target,
                             source_lang_id=src_lang,
                             target_lang_id=tgt_lang,
                             batch_size=4,
                             val_evaluators=val_metrics,
                             share_other_objective_head=tokenbsc_wiki,
                             loss_weight=25,
                             objective_id="Wiki")

seq_opensub = Sequence2Sequence(lang_module,
                                texts_or_path=opensub_pairs.source,
                                labels_or_path=opensub_pairs.target,
                                val_texts_or_path=opensub_val_pairs.source,
                                val_labels_or_path=opensub_val_pairs.target,
                                source_lang_id=src_lang,
                                target_lang_id=tgt_lang,
                                batch_size=32,
                                val_evaluators=val_metrics,
                                share_other_objective_head=tokenbsc_wiki,
                                objective_id="OpenSubtitles")

seq_bible = Sequence2Sequence(lang_module,
                              texts_or_path=bible_pairs.source,
                              labels_or_path=bible_pairs.target,
                              val_texts_or_path=bible_val_pairs.source,
                              val_labels_or_path=bible_val_pairs.target,
                              source_lang_id=src_lang,
                              target_lang_id=tgt_lang,
                              batch_size=32,
                              val_evaluators=val_metrics,
                              share_other_objective_head=tokenbsc_wiki,
                              objective_id="Bible")

schedule = ParallelSchedule(objectives=[seq_wiki, tokenbsc_wiki],
                            extra_eval_objectives=[
                                seq_opensub,
                                seq_bible
                            ],
                            args=training_arguments)

# for training from scratch, we use LangModule.reinitialize() after the initialization of all training Objectives
# lang_module.reinitialize()

adapter = Adapter(lang_module, schedule, args=training_arguments)
# train() starts the training process
adapter.train()

adapter.save_model(experiment_id)
print("Adaptation finished. Trained model for each head can be reloaded from path: `%s`" % experiment_id)

# we evaluate trained model right after the training, these should approximately match the reported results
# note that we do not perform checkpoint averaging here, that we used to report our results 
# for convenience, we average the test reports from the external logs
print("Starting evaluation")

test_device = "cuda" if torch.cuda.is_available() else "cpu"

translator_model = lang_module.trainable_models[str(id(tokenbsc_wiki))]
metric = BLEU(use_generate=True, additional_sep_char="▁", progress_bar=False)

for test_dataset_id in test_datasets:
    test_source = OPUSDataset(test_dataset_id, "test", src_lang, tgt_lang, data_dir=data_dir, firstn=test_firstn)

    references = []
    hypotheses = []
    for src_text, ref_text in zip(test_source.source, test_source.target):
        references.append(ref_text)
        inputs = lang_module.tokenizer(src_text, truncation=True, return_tensors="pt").to(test_device)

        outputs = translator_model.generate(**inputs)
        translations = lang_module.tokenizer.batch_decode(outputs, remove_special_tokens=True)
        hypotheses.append(translations[0])

    bleu = BLEU().evaluate_str(references, hypotheses)
    print("Experiment %s Test %s on %s (%s->%s): %s" %
          (experiment_id, metric, test_dataset_id, src_lang, tgt_lang, bleu))
