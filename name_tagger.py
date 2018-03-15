import pickle
import sys

import pandas as pd
import spacy

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from spacy.tokens import Doc

import settings

nlp = spacy.load('en')


class FeatureBuilder:

    def __init__(self, sent, feat):

        self.sent = sent
        self.feat = feat

        # Import existing tokenization
        self.doc = Doc(
            nlp.vocab,
            words=sent,
        )

        self.use = [

            "tok_prev",
            "tok_next",

            "is_title",

            # "is_alpha",
            # "is_ascii",
            # "is_digit",
            # "is_punct",

            # "is_lower",
            # "is_upper",

            "lemma_",
            "lower_",
            "norm_",

            "shape_",
            "prefix_",
            "suffix_",

            # "is_stop",
            # "pos_",
            # "tag_",
            # "dep_",

            # "prob",
            # "cluster",

            # "vector_norm",

        ]

        assert(len(self.sent) == len(self.doc))

        # Build features for each token in sentence
        for build in self.use:
            for tok in self.doc:
                getattr(self, build)(tok)

    """ Context features """
    def tok_prev(self, tok):
        if tok.i == 0:
            self.feat[tok.i]["tok_prev"] = "-BOS-"
        else:
            self.feat[tok.i]["tok_prev"] = self.doc[tok.i - 1].text

    def tok_next(self, tok):
        if tok.i == len(self.sent) - 1:
            self.feat[tok.i]["tok_next"] = "-EOS-"
        else:
            self.feat[tok.i]["tok_next"] = self.doc[tok.i + 1].text

    """ Ortographic features """
    def is_alpha(self, tok):
        self.feat[tok.i]["is_alpha"] = tok.is_alpha

    def is_ascii(self, tok):
        self.feat[tok.i]["is_ascii"] = tok.is_ascii

    def is_digit(self, tok):
        self.feat[tok.i]["is_digit"] = tok.is_digit

    def is_lower(self, tok):
        self.feat[tok.i]["is_lower"] = tok.is_lower

    def is_upper(self, tok):
        self.feat[tok.i]["is_upper"] = tok.is_upper

    def is_title(self, tok):
        self.feat[tok.i]["is_title"] = tok.is_title

    def is_punct(self, tok):
        self.feat[tok.i]["is_punct"] = tok.is_punct

    def like_num(self, tok):
        self.feat[tok.i]["like_num"] = tok.like_num

    """ Token features """
    def lemma_(self, tok):
        self.feat[tok.i]["lemma_"] = tok.lemma_

    def lower_(self, tok):
        self.feat[tok.i]["lower_"] = tok.lower_

    def norm_(self, tok):
        self.feat[tok.i]["norm_"] = tok.norm_

    def shape_(self, tok):
        self.feat[tok.i]["shape_"] = tok.shape_

    def prefix_(self, tok):
        self.feat[tok.i]["prefix_"] = tok.prefix_

    def suffix_(self, tok):
        self.feat[tok.i]["suffix_"] = tok.suffix_

    def is_stop(self, tok):
        self.feat[tok.i]["is_stop"] = tok.is_stop

    def pos_(self, tok):
        self.feat[tok.i]["pos_"] = tok.pos_

    def tag_(self, tok):
        self.feat[tok.i]["tag_"] = tok.tag_

    def dep_(self, tok):
        self.feat[tok.i]["dep_"] = tok.dep_

    def prob(self, tok):
        self.feat[tok.i]["prob"] = tok.prob

    def cluster(self, tok):
        self.feat[tok.i]["cluster"] = tok.cluster

    def vector_norm(self, tok):
        self.feat[tok.i]["vector_norm"] = tok.vector_norm



def build_features(split):

    if split not in ("train", "dev", "test"):
        print("Error: {0} is not a valid split, use train|dev|test")
        sys.exit(1)

    data = []

    sent = []
    feat = {}
    idx = 0

    for line in open(settings.FILE[split]["fp"], "r"):

        if not line.split():

            # Process
            fb = FeatureBuilder(sent, feat)

            for idx, tok in enumerate(sent):
                feats = fb.feat[idx]
                cols = sorted(feats.keys())
                data.append([feats[x] for x in cols])

            # Handle newline
            data.append(["-NEWLINE-"] * len(cols))

            # Reset
            sent = []
            feat = {}
            idx = 0

            #sys.exit(1)

        else:
            tok, pos, chunk, tag = line.strip().split("\t")
            feat[idx] = {
                "tok": tok,
                "pos": pos,
                "chunk": chunk,
                "tag": tag,
            }
            idx += 1
            sent.append(tok)

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(settings.FILE[split]["feat"], index=False)


def train():

    df_train = pd.read_csv(settings.FILE["train"]["feat"], keep_default_na=False)

    features = list(df_train)
    label = "tag"
    features.remove(label)

    vec = DictVectorizer()

    X_train = vec.fit_transform(df_train[features].to_dict("records"))
    y_train = df_train[label].values
    print(X_train.shape, y_train.shape)

    df_dev = pd.read_csv(settings.FILE["dev"]["feat"], keep_default_na=False)
    X_dev = vec.transform(df_dev[features].to_dict("records"))
    y_dev = df_dev[label].values
    print(X_dev.shape, y_dev.shape)

    logreg = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
    )

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_dev)

    words = df_dev["tok"].values

    with open("dev.name", "w") as out:
        for tok, tag in zip(words, y_pred):
            if (tok == "-NEWLINE-"):
                out.write("\n")
                continue

            out.write(tok + "\t" + tag + "\n")


def main():

    build_features("train")
    build_features("dev")
    # build_features("test")

    train()


if __name__ == "__main__":
    main()
