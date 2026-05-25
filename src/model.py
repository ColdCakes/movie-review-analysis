"""TF-IDF + classifier pipeline."""

from sklearn.feature_extraction.text import TfidfVectorizer  # turns review text into numbers the program can understand
from sklearn.linear_model import LogisticRegression  # the brain that learns which words mean positive and which mean negative
from sklearn.pipeline import Pipeline  # lets me connect two steps together so they run one after the other automatically
from sklearn.svm import LinearSVC  # a different kind of brain I can use instead of Logistic Regression


def build_pipeline(classifier: str = "lr") -> Pipeline:
    """
    Build a TF-IDF + classifier pipeline.

    classifier: "lr"  → Logistic Regression (default)
                "svm" → Linear SVC
    """

    tfidf = TfidfVectorizer(
        max_features=100_000,     # only pay attention to the 100,000 most common words — ignores rare typos and one-off words
        ngram_range=(1, 3),       # look at single words, pairs of words, and three word phrases so "not that good" is treated as one thing
        sublinear_tf=True,        # if a word appears many times, don't let it dominate — shrink its importance using a math trick
        min_df=2,                 # ignore any word that only shows up in one review, since it probably isn't useful
        strip_accents="unicode",  # turn letters like é or ü into plain e and u so the same word isn't counted twice
        analyzer="word",          # split the review up by words, not by individual letters
    )

    if classifier == "svm":
        clf = LinearSVC(C=1.0, max_iter=2000)  # use the alternative brain, give it up to 2000 tries to figure out the best answer
    else:
        clf = LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs")  # use the default brain, give it 1000 passes through the reviews to learn

    return Pipeline([("tfidf", tfidf), ("clf", clf)])  # connect the word-to-numbers step and the learning step into one unit so they work together
