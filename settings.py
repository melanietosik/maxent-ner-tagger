FILE = {
    # Train
    "train": {
        "fp": "CoNLL/CONLL_train.pos-chunk-name",
        "feat": "CoNLL/CONLL_train.pos-chunk-name.feat.csv",
        "tag": "CoNLL/CONLL_train.pos-chunk-name.tag.name",
    },

    # Development
    "dev": {
        "fp": "CoNLL/CONLL_dev.pos-chunk-name",
        "feat": "CoNLL/CONLL_dev.pos-chunk-name.feat.csv",
        "tag": "CoNLL/CONLL_dev.pos-chunk-name.tag.name",
    },

    # Test
    "test": {
        "fp": "CoNLL/CONLL_test.pos-chunk-name",
        "feat": "CoNLL/CONLL_test.pos-chunk-name.feat.csv",
        "tag": "CoNLL/CONLL_test.pos-chunk-name.tag.name",
    },
}

MODEL_FP = "model.pickle"
VECTORIZER_FP = "vectorizer.pickle"
