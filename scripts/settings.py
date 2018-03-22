FILE = {
    # Train
    "train": {
        "fp": "CoNLL/CONLL_train.pos-chunk-name",
        "feat": "CoNLL/CONLL_train.pos-chunk-name.feat.csv",
    },

    # Development
    "dev": {
        "fp": "CoNLL/CONLL_dev.pos-chunk-name",
        "feat": "CoNLL/CONLL_dev.pos-chunk-name.feat.csv",
        "name": "output/dev.name",
    },

    # Test
    "test": {
        "fp": "CoNLL/CONLL_test.pos-chunk-name",
        "feat": "CoNLL/CONLL_test.pos-chunk-name.feat.csv",
        "name": "output/test.name",
    },
}

MODEL_FP = "data/model.pickle"
VECTORIZER_FP = "data/vectorizer.pickle"
