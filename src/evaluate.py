"""Evaluation helpers."""

from sklearn.metrics import (
    accuracy_score,        # works out what percentage of reviews the program got right
    classification_report, # gives a detailed breakdown of how well it did on positive vs negative reviews
    confusion_matrix,      # builds a table showing exactly where the program got confused
)


def print_report(y_true, y_pred, label_names: list[str] | None = None) -> None:
    if label_names is None:  # if nobody told me what to call the two groups...
        label_names = ["negative", "positive"]  # just use negative and positive as the names

    acc = accuracy_score(y_true, y_pred)  # figure out what fraction of reviews were labeled correctly
    print(f"\nAccuracy : {acc:.4f}  ({acc * 100:.2f}%)")  # show that as both a decimal and a percentage
    print("\nClassification Report:")  # print a title before the big table
    print(classification_report(y_true, y_pred, target_names=label_names))  # print the detailed results table

    cm = confusion_matrix(y_true, y_pred)  # create the grid that shows correct and incorrect predictions
    print("Confusion Matrix (rows=actual, cols=predicted):")  # print a title for the grid
    header = f"{'':12}" + "  ".join(f"{n:>10}" for n in label_names)  # build the top row of the grid with the column names
    print(header)  # print that top row
    for name, row in zip(label_names, cm):  # go through the grid one row at a time
        cells = "  ".join(f"{v:>10}" for v in row)  # lay out the numbers in that row neatly
        print(f"{name:<12}{cells}")  # print the row name and its numbers side by side


def top_features(pipeline, n: int = 20) -> None:
    """Print the top n TF-IDF features for each class (LR only)."""
    try:
        tfidf = pipeline.named_steps["tfidf"]          # reach into the pipeline and get the word-to-numbers part
        clf = pipeline.named_steps["clf"]              # reach in and get the learning brain part
        feature_names = tfidf.get_feature_names_out()  # get the full list of words and phrases the program knows about
        coefs = clf.coef_[0]                           # get the score the program gave to every single word
    except (AttributeError, IndexError):
        return  # if something goes wrong here just skip this quietly and move on

    import numpy as np  # a tool I need just for sorting the scores

    top_pos = coefs.argsort()[-n:][::-1]  # find the words with the highest positive scores
    top_neg = coefs.argsort()[:n]         # find the words with the lowest (most negative) scores

    print(f"\nTop {n} positive-sentiment features:")  # print a title
    for i in top_pos:  # go through each strongly positive word
        print(f"  {feature_names[i]:<30}  {coefs[i]:+.4f}")  # print the word and the score next to it

    print(f"\nTop {n} negative-sentiment features:")  # print a title
    for i in top_neg:  # go through each strongly negative word
        print(f"  {feature_names[i]:<30}  {coefs[i]:+.4f}")  # print the word and the score next to it
