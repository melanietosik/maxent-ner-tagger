# Maximum-entropy NER tagger

A [GitHub repository for this project](https://github.com/melanietosik/maxent-ner-tagger) is available online.

## Overview

The goal of this project was to implement and train a [named-entity recognizer (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition). Most of the feature builder functionality was implemented using [spaCy](https://spacy.io/), an industrial-strength, open-source NLP library written in Python/Cython. For classification a maximum entropy (MaxEnt) classifier is used.

## Implementation details

The dataset for this task is the [2003 CoNLL (Conference on Natural Language Learning)](https://aclweb.org/anthology/W/W03/W03-0419.pdf) corpus, which is primarily composed of Reuters news data. The data files are pre-preprocessed and already contain one token per line, its part-of-speech (POS) tag, a BIO (short for beginning, inside, outside) chunk tag, and the corresponding NER tag. 

SpaCy's built-in [token features](https://spacy.io/api/token) proved most useful for feature engineering. Making use of external word lists, such as the Wikipedia gazetteers distributed as part of the [Illinois Named Entity Tagger](https://cogcomp.org/page/software_view/NETagger), generally lead to a decrease in tagging accuracy. Since the data files are relatively large, the gazetteer source code and files are not included in the final submission. I also experimented with boosting model performance by including the prior state/tag as a feature. Somewhat surprisingly, model performance largely remained unchanged, presumably due to the fact that each label is predicted from the same feature set that is encoded in the model anyway.

A [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) classifier is used for classification. Specifically, I used the ["Logistic Regression" classifier implemented in scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). For a proof of mathematical equivalency, please see ["The equivalence of logistic regression and maximum entropy models"](http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf) (Mount, 2011).

To make I/O processes a little more uniform, I converted the original data files for the training, development, and test split to a single `.pos-chunk-name` file per split, each containing one token per line and the POS, BIO, and name tags. The conformed data files can be found in the [`CoNLL/`](https://github.com/melanietosik/maxent-ner-tagger/tree/master/CoNLL) data directory.

## Running the code

The NER tagger is implemented in [`scripts/name_tagger.py`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/scripts/name_tagger.py). With each run, the programm will extract features for the training, development, and test split. All feature files are written to the [`CoNNL/`](https://github.com/melanietosik/maxent-ner-tagger/tree/master/CoNLL) data directory with a `.feat.csv` extension. The `.csv` files are compact and readable and can be manually inspected.

After model training, the program also writes a serialized model and vectorizer object to the [`data/`](https://github.com/melanietosik/maxent-ner-tagger/tree/master/data) directory. The vectorizer object is a pickled dump of scikit-learn's ["DictVectorizer"](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) fit to the training data, which is used to transform the lists of feature-value mappings to binary (one-hot) vectors. In other words, one boolean-valued feature is constructed for each of the possible string values that a feature can take on. For example, a feature ``is_stop`` that can either be `"True"` or `"False"` will become two features in the output, one for `"is_stop=True"` and one for `"is_stop=False"`.

Finally, the program writes two tagged `.name` files to the [`output/`](https://github.com/melanietosik/maxent-ner-tagger/tree/master/output) directory, one for the development set ([`dev.name`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/output/dev.name)) and one for the test set ([`test.name`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/output/test.name)). All settings can be adjusted by editing the paths specified in [`scripts/settings.py`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/scripts/settings.py).

### Install requiremements

Before running the code, you need to create a [virtual environment](https://virtualenv.pypa.io/en/stable/) and install the required Python dependencies:

```
[maxent-ner-tagger]$ virtualenv -p python3 env
[maxent-ner-tagger]$ source env/bin/activate
[maxent-ner-tagger]$ pip install -U spacy
[maxent-ner-tagger]$ python -m spacy download en  # Download spaCy's English language model files
[maxent-ner-tagger]$ pip install -r requirements.txt
```

### Run the tagger

To **(re-)run the tagger**, in the root directory of the project, run:

```
(env) [maxent-ner-tagger]$ python scripts/name_tagger.py
```

You should start seeing output pretty much immediately. It takes about 5 minutes to regenerate the features and retrain the model. Please note that **all output files will be over-written** with each run.

```
(env) [maxent-ner-tagger]$ python scripts/name_tagger.py
Generating train features...
Lines processed:        0
Lines processed:    10000
Lines processed:    20000
Lines processed:    30000
Lines processed:    40000
Lines processed:    50000
Lines processed:    60000
Lines processed:    70000
Lines processed:    80000
Lines processed:    90000
Lines processed:   100000
Lines processed:   110000
Lines processed:   120000
Lines processed:   130000
Lines processed:   140000
Lines processed:   150000
Lines processed:   160000
Lines processed:   170000
Lines processed:   180000
Lines processed:   190000
Lines processed:   200000
Lines processed:   210000

Writing output file: CoNLL/CONLL_train.pos-chunk-name.feat.csv

Generating dev features...
Lines processed:        0
Lines processed:    10000
Lines processed:    20000
Lines processed:    30000
Lines processed:    40000
Lines processed:    50000

Writing output file: CoNLL/CONLL_dev.pos-chunk-name.feat.csv

Generating test features...
Lines processed:        0
Lines processed:    10000
Lines processed:    20000
Lines processed:    30000
Lines processed:    40000
Lines processed:    50000

Writing output file: CoNLL/CONLL_test.pos-chunk-name.feat.csv

Training model...
X (219554, 449494)
y (219554,)

Tagging dev...
X (55044, 449494)
y (55044,)

Writing output file: output/dev.name

Tagging test...
X (50350, 449494)
y (50350,)

Writing output file: output/test.name

Scoring development set...
50523 out of 51578 tags correct
  accuracy: 97.95
5917 groups in key
6006 groups in response
5160 correct groups
  precision: 85.91
  recall:    87.21
  F1:        86.56

python scripts/name_tagger.py  307.34s user 20.51s system 105% cpu 5:11.00 total
```

## Results

By default, the model always makes use of the POS and BIO tags that are provided with the CoNLL data sets. In addition, I experimented with the following groups of token features:

- `idx`
- `is_alpha`, `is_ascii`, `is_digit`, `is_punct`, `like_num`
- `is_title`, `is_lower`, `is_upper`
- `orth_`, `lemma_`, `lower_`, `norm_`
- `shape_`
- `prefix_`, `suffix_`
- `pos_`, `tag_`, `dep_`
- `is_stop`, `cluster`
- `head`, `left_edge`, `right_edge`

Most of features should be self-explanatory. For detailed descriptions, please see the documentation in [`scripts/name_tagger.py`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/scripts/name_tagger.py) or spaCy's [Token API documentation](https://spacy.io/api/token). Since only one sentence is processed at a time, I decided to also include _context token_ features. Currently, the model uses the token features of `n=2` tokens to the left and the right of the target token.

**Note:** I slightly modified the `scorer.name.py` script so it could be easily be imported as a module and run on Python 3. No changes were made to the actual scoring functionality. The revised script is stored in [`scripts/scorer.py`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/scripts/scorer.py).

### Development set

A breakdown of the experimental feature sets and corresponding _group accuracies_ on the development set is provided below. For the full log of results, please see [`log.md`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/log.md).

| Feature set                                                                                                                                |                            Accuracy |
|:-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------:|
| CoNLL only                                                                                                                                 | P: 66.77<br> R: 71.46<br> **F1: 69.03** |
| CoNLL + `is_title`, `orth_`                                                                                                                | P: 70.93<br> R: 72.74<br> **F1: 71.82** |
| CoNLL + `idx`, `is_title`, `orth_`                                                                                                         | P: 69.45<br> R: 72.84<br> **F1: 71.10** |
| CoNLL + `is_alpha`, `is_ascii`, `is_digit`, `is_punct`, `is_title`, `like_num`, `orth_`                                                    | P: 68.66<br> R: 70.36<br> **F1: 69.50** |
| CoNLL + `is_lower`, `is_title`, `is_upper`, `orth_`                                                                                        | P: 69.99<br> R: 73.79<br> **F1: 71.84** |
| CoNLL + `is_title`, `lemma_`, `orth_`                                                                                                      | P: 73.17<br> R: 76.14<br> **F1: 74.62** |
| CoNLL + `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`                                                                                   | P: 73.37<br> R: 76.83<br> **F1: 75.06** |
| CoNLL + `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `shape_`                                                                         | P: 72.94<br> R: 78.77<br> **F1: 75.75** |
| CoNLL + `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `prefix_`, `shape_`, `suffix_`                                                   | P: 71.51<br> R: 78.54<br> **F1: 74.86** |
| CoNLL + `dep_`, `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `pos_`, `shape_`, `tag_`                                                 | P: 72.89<br> R: 79.80<br> **F1: 76.19** |
| CoNLL + `cluster`, `dep_`, `is_stop`, `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `pos_`, `shape_`, `tag_`                           | P: 70.32<br> R: 78.38<br> **F1: 74.13** |
| CoNLL + `dep_`, `head`, `is_title`, `left_edge`, `lemma_`, `lower_`, `norm_`, `orth_`, `pos_`, `right_edge`, `shape_`, `tag_`              | P: 78.76<br> R: 83.37<br> **F1: 81.00** |
| CoNLL + `dep_`, `head`, `is_title`, `left_edge`, `lemma_`, `lower_`, `norm_`, `orth_`, `pos_`, `right_edge`, `shape_`, `tag_` + context    | P: 85.64<br> R: 87.07<br> **F1: 86.35** |
| CoNLL + `head`, `is_title`, `left_edge`, `lemma_`, `lower_`, `norm_`, `orth_`, `right_edge`, `shape_` + context                            | P: 85.90<br> R: 86.92<br> **F1: 86.41** |
| CoNLL + `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `shape_`] + context                                                              | P: 85.75<br> R: 86.51<br> **F1: 86.13** |
| **CoNLL + `is_title`, `lemma_`, `lower_`, `norm_`, `orth_`, `prefix_`, `shape_`, `suffix_` + context**                                       | P: 85.91<br> R: 87.21<br> **F1: 86.56** |
| CoNLL + `is_title`, `lemma_`, `orth_`, `prefix_`, `shape_`, `suffix_` + context                                                            | P: 85.80<br> R: 86.99<br> **F1: 86.39** |

Overall, I found that the word itself and its various normalized word forms (exact, lemma, lowercased, normalized) were very helpful indicators for NER tagging. In addition, including whether or not the token was written in title case and its general shape (e.g. `Xxxx` or `dd`) improved model performance as well. Finally, I was able to achieve a **F1 score of 86.56** on the development set by incorporating context token features as well.

### Test set

The tagged output file for the test is written to [`output/test.name`](https://github.com/melanietosik/maxent-ner-tagger/blob/master/output/test.name). To score the test file directly after model training, make sure the tagged gold file exists at `CoNLL/CONLL_test.name` and comment in lines `402-404` in `scripts/name_tagger.py`. Otherwise you can just use the original scorer script directly with the tagged output file.

