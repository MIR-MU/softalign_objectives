import comet_ml  # logging hook must be imported before torch # noqa F401
import torch

from adaptor.adapter import Adapter
from adaptor.evaluators.generative import BLEU, ROUGE, BERTScore
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import AdaptationArguments, StoppingStrategy
from utils.data_utils_opus import OPUSDataset, OPUS_RESOURCES_URLS

data_dir = "utils"
experiment_id = "lora_dgt"

src_lang = "de"
tgt_lang = "en"

# 1. Load OPUS domain-specific data sets
train_firstn = None  # no limit
val_firstn = 500
test_firstn = 1000


train_dataset_id = "Bible"
# we test on all the domains in the constructed collection
test_dataset_ids = OPUS_RESOURCES_URLS.keys()

# reordering of the data sets gives priority to the first one in deduplication
val_dataset = OPUSDataset(train_dataset_id, "val", src_lang, tgt_lang, data_dir=data_dir, firstn=val_firstn)
train_dataset = OPUSDataset(train_dataset_id, "train", src_lang, tgt_lang, data_dir=data_dir, firstn=train_firstn)

# 2. Initialize training arguments
# We apply NUM_STEPS stopping strategy in cases where at least one of the objectives does not converge in max_steps
training_arguments = AdaptationArguments(output_dir=experiment_id,
                                         learning_rate=2e-4,
                                         # stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         stopping_strategy=StoppingStrategy.NUM_STEPS_ALL_OBJECTIVES,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=400000,
                                         gradient_accumulation_steps=3,
                                         logging_steps=50,
                                         eval_steps=500,
                                         save_steps=5000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps",
                                         also_log_converged_objectives=True)

# we initialise base model from HF model
lang_module = LangModule("Helsinki-NLP/opus-mt-%s-%s" % (src_lang, tgt_lang))

metrics_args = {"additional_sep_char": "▁"}

val_metrics = [BLEU(**metrics_args, decides_convergence=True), ROUGE(**metrics_args), BERTScore(**metrics_args)]

# training objectives:
# validations are also computed by the training MLE objective
train_mle = Sequence2Sequence(lang_module,
                              texts_or_path=train_dataset.source,
                              labels_or_path=train_dataset.target,
                              val_texts_or_path=val_dataset.source,
                              val_labels_or_path=val_dataset.target,
                              source_lang_id=src_lang,
                              target_lang_id=tgt_lang,
                              batch_size=10,
                              val_evaluators=val_metrics,
                              objective_id=train_dataset_id)

training_objectives = [train_mle]

test_datasets = []
test_objectives = []

for dataset_id in test_dataset_ids:
    if dataset_id == train_dataset_id:
        # train domain evaluated by train evaluator; deduplication would make a new objective with empty data
        continue

    test_dataset = OPUSDataset(dataset_id, "val", src_lang, tgt_lang, data_dir=data_dir, firstn=test_firstn)
    train_dataset = OPUSDataset(dataset_id, "train", src_lang, tgt_lang, data_dir=data_dir, firstn=train_firstn)
    test_datasets.append(test_dataset)

    new_eval_objective = Sequence2Sequence(lang_module,
                                           texts_or_path=train_dataset.source,
                                           labels_or_path=train_dataset.source,
                                           val_texts_or_path=test_dataset.source,
                                           val_labels_or_path=test_dataset.target,
                                           source_lang_id=src_lang,
                                           target_lang_id=tgt_lang,
                                           batch_size=10,
                                           val_evaluators=val_metrics,
                                           share_other_objective_head=train_mle,
                                           objective_id="%s-%s" % (test_dataset.split, test_dataset.domain_label))

    test_objectives.append(new_eval_objective)

schedule = ParallelSchedule(objectives=training_objectives,
                            extra_eval_objectives=test_objectives,
                            args=training_arguments)

adapter = Adapter(lang_module, schedule, args=training_arguments)

patch_linears_with_lora(lang_module, r=16)

lang_module.train()
loralib.mark_only_lora_as_trainable(lang_module)

trainables = {n: p for n, p in lang_module.named_parameters() if p.requires_grad}
loras = {n: p for n, p in lang_module.named_parameters() if "lora_" in n}
assert set(loras.keys()) == set(trainables.keys()), "Only LoRa weights should be set to be trained"

adapter.train()

adapter.save_model(experiment_id)
print("Adaptation finished. Trained model for each head can be reloaded from path: `%s`" % experiment_id)

# we evaluate trained model right after the training, these should approximately match the reported results
# note that we do not perform checkpoint averaging here, that we used to report our results 
# for convenience, we average the test reports from the external logs
print("Starting evaluation")

test_device = "cuda" if torch.cuda.is_available() else "cpu"

translator_model = lang_module.trainable_models[str(id(train_mle))]

for test_dataset in test_datasets:

    references = []
    hypotheses = []
    for src_text, ref_text in zip(test_dataset.source, test_dataset.target):
        references.append(ref_text)
        inputs = lang_module.tokenizer(src_text, truncation=True, return_tensors="pt").to(test_device)

        outputs = translator_model.generate(**inputs)
        translations = lang_module.tokenizer.batch_decode(outputs, remove_special_tokens=True)
        hypotheses.append(translations[0])

    from datetime import datetime

    translations_file = "%s-%s.txt" % (experiment_id, datetime.now().strftime("%m%d-%H%M%S"))
    with open(translations_file, "w") as out_f:
        print("Writing translations to %s" % translations_file)
        out_f.writelines(hypotheses)

    metrics_values = {str(metric): metric.evaluate_str(references, hypotheses) for metric in val_metrics}
    print("Experiment %s Test metrics on %s (%s->%s) test scores: %s" %
          (experiment_id, test_dataset.domain_label, src_lang, tgt_lang, metrics_values))
