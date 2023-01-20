## Experiments running scripts

This folder contains reproducible scripts for all our experiments. 
Each experimental script should be self-explaining and reproducible as-is, without any additional parameters.

We have run our experiments from the project root, in this exact format:

```shell
CUDA_VISIBLE_DEVICES=XX COMET_PROJECT_NAME={your-project} COMET_API_KEY={key} python experiments/seq_align/adapt_emea_es-en.py
```

The experiments are categorized into folders and files 
by (1) training objective and (2) by the training domain:

1. Main new objectives:
* `token_align` contain scripts resulting in the reported *TokenAlign* evaluations.
* `seq_align` contain scripts resulting in the reported *SeqAlign* evaluations.

2. Ablation objectives:
* `ablations/SCE` contain scripts resulting in the additional ablation results reported under *SCE* evaluations.
* `ablations/seq_align_dec` contain scripts resulting in the additional ablation results reported under *SeqAlign-dec* evaluations.
* `ablations/seq_random` contain scripts resulting in the additional ablation results reported under *SCE* evaluations.

3. Baselines:
* `baselines/adapters` contain scripts of experiments with Adapters of Houlsby (2019) 
* `baselines/ensembling` contain scripts of experiments using the ensemble of pre-trained and fine-tuned models (Freitag, 2016), adapted to Transformers 
* `baselines/lora` contain scripts of experiments using LoRA of Hu (2022)
* `baselines/lora` contain scripts of experiments of smoothed MLE (Szegedy, 2016)

### Selected domains:

All our objectives are evaluated on the following domains ordered by size.

All our training and evaluation domains are publicly available on https://opus.nlpl.eu.
We also deliver unified interface for their access: see `utils/data_utils/OPUSDataset` and its use within the experiments.

1. bible: german -> english: 2.9M tokens, 0.06 sentences

2. TEDTalks: english -> chinese (simplified): 5.1M tokens, 0.1M sentences

3. opensub: english -> ukraine: 9.5M tokens 0.8M sentences

4. wikipedia: english -> czech: 4.3M tokens, 0.1M sentences

5. EMEA v3: estonian -> english: 5.7M tokens, 0.3M sentences

6. DGT: english -> deutsch: 166M words, 5.1M sentences

