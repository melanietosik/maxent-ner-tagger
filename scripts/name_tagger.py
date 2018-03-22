import pickle
import sys

import pandas as pd
import spacy

import score
import settings

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from spacy.tokens import Doc

nlp = spacy.load('en')


class FeatureBuilder:

    def __init__(self, sent, feat):

        self.sent = sent  # Sentence
        self.feat = feat  # Features

        # Import existing tokenization
        self.doc = Doc(
            nlp.vocab,
            words=sent,
        )

        # Add token annotations
        nlp.tagger(self.doc)
        nlp.parser(self.doc)

        # List of additional features to use
        self.use = [
            # "idx",

            # "is_alpha",
            # "is_ascii",
            # "is_digit",
            # "is_punct",
            # "like_num",

            "is_title",
            # "is_lower",
            # "is_upper",

            "orth_",
            "lemma_",
            "lower_",
            "norm_",

            "shape_",

            "prefix_",
            "suffix_",

            # "pos_",
            # "tag_",
            # "dep_",

            # "is_stop",
            # "cluster",

            # "head",
            # "left_edge",
            # "right_edge",
        ]

        assert(len(self.sent) == len(self.doc))

        # Build token features
        for x in self.use:
            for tok in self.doc:
                getattr(self, x)(tok)

        # Add context token features
        for tok in self.doc:
            for x in self.use:
                # Token - 1
                if tok.i == 0:
                    self.feat[tok.i]["prev-1_" + x] = "-BOS-"
                else:
                    self.feat[tok.i]["prev-1_" + x] = self.feat[tok.i - 1][x]
                # Token - 2
                if (tok.i == 0) or (tok.i == 1):
                    self.feat[tok.i]["prev-2_" + x] = "-BOS-"
                else:
                    self.feat[tok.i]["prev-2_" + x] = self.feat[tok.i - 2][x]
                # Token + 1
                if tok.i == len(self.sent) - 1:
                    self.feat[tok.i]["next+1_" + x] = "-EOS-"
                else:
                    self.feat[tok.i]["next+1_" + x] = self.feat[tok.i + 1][x]
                # Token + 2
                if (tok.i == len(self.sent) - 2) or (tok.i == len(self.sent) - 1):
                    self.feat[tok.i]["next+2" + x] = "-EOS-"
                else:
                    self.feat[tok.i]["next+2" + x] = self.feat[tok.i + 2][x]

    def idx(self, tok):
        """
        [spaCy] The character offset of the token within the parent document.
        """
        self.feat[tok.i]["idx"] = tok.i

    def is_alpha(self, tok):
        """
        [spaCy] Does the token consist of alphabetic characters?
        """
        self.feat[tok.i]["is_alpha"] = tok.is_alpha

    def is_ascii(self, tok):
        """
        [spaCy] Does the token consist of ASCII characters?
        """
        self.feat[tok.i]["is_ascii"] = tok.is_ascii

    def is_digit(self, tok):
        """
        [spaCy] Does the token consist of digits?
        """
        self.feat[tok.i]["is_digit"] = tok.is_digit

    def is_punct(self, tok):
        """
        [spaCy] Is the token punctuation?
        """
        self.feat[tok.i]["is_punct"] = tok.is_punct

    def like_num(self, tok):
        """
        [spaCy] Does the token represent a number? e.g. "10.9", "10", "ten", etc.
        """
        self.feat[tok.i]["like_num"] = tok.like_num

    def is_title(self, tok):
        """
        [spaCy] Is the token in titlecase?
        """
        self.feat[tok.i]["is_title"] = tok.is_title

    def is_lower(self, tok):
        """
        [spaCy] Is the token in lowercase?
        """
        self.feat[tok.i]["is_lower"] = tok.is_lower

    def is_upper(self, tok):
        """
        [spaCy] Is the token in uppercase?
        """
        self.feat[tok.i]["is_upper"] = tok.is_upper

    def orth_(self, tok):
        """
        [spaCy] Verbatim text content.
        """
        self.feat[tok.i]["orth_"] = tok.orth_

    def lemma_(self, tok):
        """
        [spaCy] Base form of the token, with no inflectional suffixes.
        """
        self.feat[tok.i]["lemma_"] = tok.lemma_

    def lower_(self, tok):
        """
        [spaCy] Lowercase form of the token text.
        """
        self.feat[tok.i]["lower_"] = tok.lower_

    def norm_(self, tok):
        """
        [spaCy] The token's norm, i.e. a normalised form of the token text.
        """
        self.feat[tok.i]["norm_"] = tok.norm_

    def shape_(self, tok):
        """
        [spaCy] Transform of the tokens's string, to show orthographic features. For example, "Xxxx" or "dd".
        """
        self.feat[tok.i]["shape_"] = tok.shape_

    def cluster(self, tok):
        """
        [spaCy] Brown cluster ID.
        """
        self.feat[tok.i]["cluster"] = tok.cluster

    def prefix_(self, tok):
        """
       [spaCy] A length-N substring from the start of the token. Defaults to N=1.
        """
        self.feat[tok.i]["prefix_"] = tok.prefix_

    def suffix_(self, tok):
        """
        [spaCy] Length-N substring from the end of the token. Defaults to N=3.
        """
        self.feat[tok.i]["suffix_"] = tok.suffix_

    def pos_(self, tok):
        """
        [spaCy] Coarse-grained part-of-speech.
        """
        self.feat[tok.i]["pos_"] = tok.pos_

    def tag_(self, tok):
        """
        [spaCy] Fine-grained part-of-speech.
        """
        self.feat[tok.i]["tag_"] = tok.tag_

    def dep_(self, tok):
        """
        [spaCy] Syntactic dependency relation.
        """
        self.feat[tok.i]["dep_"] = tok.dep_

    def is_stop(self, tok):
        """
        [spaCy] Is the token part of a "stop list"?
        """
        self.feat[tok.i]["is_stop"] = tok.is_stop

    def head(self, tok):
        """
        [spaCy] The syntactic parent, or "governor", of this token.
        """
        self.feat[tok.i]["head"] = tok.head.text

    def left_edge(self, tok):
        """
        [spaCy] The leftmost token of this token's syntactic descendants.
        """
        self.feat[tok.i]["left_edge"] = tok.left_edge.text

    def right_edge(self, tok):
        """
        [spaCy] The rightmost token of this token's syntactic descendents.
        """
        self.feat[tok.i]["right_edge"] = tok.right_edge.text


def build_features(split):
    """
    Helper function for feature extraction
    """
    if split not in ("train", "dev", "test"):
        print("Error: {0} is not a valid split, use train|dev|test")
        sys.exit(1)

    print("Generating {0} features...".format(split))

    data = []
    sent = []
    feat = {}
    idx = 0

    for cnt, line in enumerate(open(settings.FILE[split]["fp"], "r")):

        # Process sentence
        if not line.split():

            # Extract features
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

        else:
            tok, pos, chunk, tag = line.strip().split("\t")
            feat[idx] = {
                "tok": tok,
                "pos": pos,
                "chunk": chunk,
                "tag": tag,
            }
            idx += 1
            # Keep appending to current sentence
            sent.append(tok)

        if cnt % 10000 == 0:
            print("Lines processed: {:>8}".format(cnt))

    print("\nWriting output file: {0}\n".format(settings.FILE[split]["feat"]))
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(settings.FILE[split]["feat"], index=False)


def train():
    """
    Train and write MaxEnt model
    """
    df = pd.read_csv(settings.FILE["train"]["feat"], keep_default_na=False)

    # Get list of features
    features = list(df)

    # Get target label
    label = "tag"
    features.remove(label)

    # Feature vectorizer
    vec = DictVectorizer()

    # Fit and transform training data
    X_train = vec.fit_transform(df[features].to_dict("records"))
    y_train = df[label].values

    print("Training model...")
    print("X", X_train.shape)
    print("y", y_train.shape)
    print()

    logreg = LogisticRegression(
        multi_class="multinomial",  # Use cross-entropy loss
        solver="lbfgs",             # Use limited-memory BFGS (L-BFGS) optimizer
        C=2.0,                      # Set inverse of regularization strength
        n_jobs=-1,                  # Parallelize if possible
    )

    # Fit model to training data
    logreg.fit(X_train, y_train)

    with open(settings.MODEL_FP, "wb") as model_file:
        pickle.dump(logreg, model_file)

    with open(settings.VECTORIZER_FP, "wb") as vectorizer_file:
        pickle.dump(vec, vectorizer_file)


def tag(split):
    """
    Tag development or test data
    """
    df = pd.read_csv(settings.FILE[split]["feat"], keep_default_na=False)

    features = list(df)
    label = "tag"
    features.remove(label)

    model = pickle.load(open(settings.MODEL_FP, "rb"))
    vec = pickle.load(open(settings.VECTORIZER_FP, "rb"))

    # Get features and labels
    X = vec.transform(df[features].to_dict("records"))
    y = df[label].values

    print("Tagging {0}...".format(split))
    print("X", X.shape)
    print("y", y.shape)

    # Predicted tags
    y_pred = model.predict(X)

    # Get list of tokens
    toks = df["tok"].values

    # Write output file
    print("\nWriting output file: {0}\n".format(settings.FILE[split]["name"]))
    with open(settings.FILE[split]["name"], "w") as out:
        for tok, tag in zip(toks, y_pred):
            # Handle newlines
            if (tok == "-NEWLINE-"):
                out.write("\n")
                continue
            out.write(tok + "\t" + tag + "\n")


def main():
    """
    main()
    """
    # Build features for each data split
    for split in ("train", "dev", "test"):
        build_features(split)

    # Train model
    train()

    # Tag dev and test split
    for split in ("dev", "test"):
        tag(split)

    # Get score on dev data
    print("Scoring development set...")
    score.score("CoNLL/CONLL_dev.name", settings.FILE["dev"]["name"])

    # # Get score on test data
    # print("Scoring test set...")
    # score.score("CoNLL/CONLL_test.name", settings.FILE["test"]["name"])


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print("Usage: python name_tagger.py")
        sys.exit(1)

    main()
