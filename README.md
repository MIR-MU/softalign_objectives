# Soft Alignment Objectives for Robust Adaptation in Machine Translation

This repository contains implementations of soft alignment objectives proposed in paper *Soft Alignment Objectives for Robust Adaptation in Machine Translation*
and the reproducible scripts for collecting the results reported in the paper.

For easier readability and reproducibility build our objectives and experiments on top of the objective-centric 
[Adaptor](https://github.com/gaussalgo/adaptor) library, integrated on top of HuggingFace Transformers.

While this fork of Adaptor is required to run the reproduction scripts, the resulting models can be used in standalone
as any other HuggingFace model -- see [the example](https://github.com/gaussalgo/adaptor#adapted-machine-translation).

1. **Objectives**: We implement our main and ablation objectives in [new_objectives folder](adaptor/new_objectives).
Importantly, each objective instance implements its `_compute_loss()` method returning a scalar with grad_fn used to update
the model.
2. **Experiments**: The running scripts of all our experiments can be found in [experiments](experiments) folder. 
See its respective readme for more information.

**Other requirements**

Before running the experiments, please run
```shell
# activate your dedicated virtual enviromnent
git clone {this repo}
cd soft_mt_adaptation
pip install -e .[generative]
```

Then you can choose the configuration and run it from the project root, for instance:
```shell
CUDA_VISIBLE_DEVICES=XX python experiments/seq_align/adapt_wiki_en-cs.py
```
The results will be collected in your configured logging environment. 
We used [comet.ml](https://comet.ml) for automated logging collection.

We report the configuration of our environmental testbed in our paper:
```text
We performed the adaptation of each of the proposed objectives on a server with a single NVidia Tesla A100, 
80 GB of graphic memory, 512 GB of1013 RAM and 64-Core Processor (AMD EPYC 7702P). 
We also tested to train all our experiments using lower configuration using 
a single NVidia Tesla T4, 16 GB of graphic memory, 20 GB of RAM and a single core of Intel(R) Xeon(R) processor.
```

**Need further help?** File an issue in the project repository and we will take a look! 

