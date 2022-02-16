from adaptor.utils import AdaptationArguments, StoppingStrategy

paths = {
    "texts": {
        "ner": "mock_data/supervised_texts.txt",
        "classification": "mock_data/supervised_texts.txt",
        "translation": "mock_data/seq2seq_sources.txt",
        "unsup": "mock_data/domain_unsup.txt"
    },
    "labels": {
        "ner": "mock_data/supervised_texts_token_labels.txt",
        "classification": "mock_data/supervised_texts_sequence_labels.txt",
        "translation": "mock_data/seq2seq_targets.txt"
    }
}

test_base_models = {
    "translation_mono": "Helsinki-NLP/opus-mt-en-cs",
    "translation_multi": {
            "model": "sshleifer/tiny-mbart",
            "test_src_lang": "en_XX",
            "test_tgt_lang": "cs_CZ"},
    "token_classification": "bert-base-multilingual-cased",
    "sequence_classification": "bert-base-multilingual-cased"
}

training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.FIRST_OBJECTIVE_NUM_EPOCHS,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=2,
                                         log_level="critical",
                                         logging_steps=1,
                                         num_train_epochs=2)
