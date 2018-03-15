import pickle
import sys

import numpy as np
import pandas as pd
import spacy

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import settings

nlp = spacy.load('en')


class FeatureBuilder:

    def __init__(self, token):
        self.token = token
        self.feats = {}
        self.names = [
            # "length",
            "first_is_upper",
            "has_vector",

        ]
        self.sptoken = nlp(token)

        for feat in self.names:
            getattr(self, feat)()

        self.names.sort(key=lambda x: x[1])

    def length(self):
        self.feats["length"] = len(self.token)

    def first_is_upper(self):
        self.feats["first_is_upper"] = self.token[0].isupper()

    def has_vector(self):
        self.feats["has_vector"] = self.sptoken.vector_norm


def build_features(split):

    if split not in ("train", "dev", "test"):
        print("Error: {0} is not a valid split, use train|dev|test")
        sys.exit(1)

    data = []

    for line in open(settings.FILE_MAP[split]["fp"], "r"):

        if not line.strip():
            tok = pos = chunk = tag = "--n--"
        else:
            tok, pos, chunk, tag = line.strip().split("\t")

        fb = FeatureBuilder(tok)

        data.append([tok, pos, chunk] + [fb.feats[k] for k in fb.names] + [tag])

    # Pandas
    columns = ["tok", "pos", "chunk"] + fb.names + ["tag"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(settings.FILE_MAP[split]["feat"], index=False)


def train():

    df_train = pd.read_csv(settings.FILE_MAP["train"]["feat"], keep_default_na=False)

    features = list(df_train)
    label = "tag"
    features.remove(label)

    vec = DictVectorizer()

    X_train = vec.fit_transform(df_train[features].to_dict("records"))
    y_train = df_train[label].values
    print(X_train.shape, y_train.shape)

    df_dev = pd.read_csv(settings.FILE_MAP["dev"]["feat"], keep_default_na=False)
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

    with open("out.name", "w") as out:
        for tok, tag in zip(words, y_pred):
            if (tok == "--n--"):
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
