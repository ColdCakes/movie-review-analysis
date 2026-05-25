"""IMDB dataset loading and preprocessing."""

import re  # a tool that lets me find and replace things inside text
from datasets import load_dataset  # a tool that downloads the IMDB reviews for me automatically


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)           # remove hidden website code that sometimes appears in reviews, like <br />
    text = re.sub(r"[^a-zA-Z\s]", " ", text)        # remove anything that isn't a letter, like numbers and punctuation
    text = re.sub(r"\s+", " ", text).strip().lower() # remove extra spaces and turn all letters into lowercase
    return text  # give back the cleaned up review


def _balanced_select(split, n: int):
    """Select n examples with equal positive/negative representation."""
    import numpy as np  # a tool I need here just for the random shuffling
    labels = split["label"]  # get the positive or negative marking for every review
    per_class = n // 2  # split the total number in half so I take the same amount from each group
    pos_idx = [i for i, l in enumerate(labels) if l == 1][:per_class]  # find the position of each positive review, only up to my limit
    neg_idx = [i for i, l in enumerate(labels) if l == 0][:per_class]  # find the position of each negative review, only up to my limit
    indices = neg_idx + pos_idx  # put both lists together into one
    np.random.shuffle(indices)  # mix them up so they aren't all negative first then all positive
    return split.select(indices)  # give back only the reviews at those positions


def load_imdb(max_train: int | None = None, max_test: int | None = None):
    """Return (X_train, y_train, X_test, y_test) as plain Python lists."""
    print("Loading IMDB dataset...")  # show a message so the user knows something is happening
    ds = load_dataset("stanfordnlp/imdb")  # go and download all 50,000 IMDB reviews

    train = ds["train"]  # set aside the 25,000 reviews the program will learn from
    test = ds["test"]  # set aside the 25,000 reviews I'll use to test how well it learned

    if max_train:  # if the user asked for fewer training reviews...
        train = _balanced_select(train, max_train)  # cut it down but keep an equal mix of positive and negative
    if max_test:  # if the user asked for fewer test reviews...
        test = _balanced_select(test, max_test)  # cut that down too, keeping it balanced

    print(f"  Train samples : {len(train)}")  # show how many reviews the program will learn from
    print(f"  Test  samples : {len(test)}")  # show how many reviews will be used for testing

    X_train = [clean_text(t) for t in train["text"]]  # clean up every single training review
    y_train = train["label"]  # keep track of whether each training review is positive or negative

    X_test = [clean_text(t) for t in test["text"]]  # clean up every single test review
    y_test = test["label"]  # keep track of whether each test review is positive or negative

    return X_train, y_train, X_test, y_test  # hand all four things back to the part of the program that asked for them
