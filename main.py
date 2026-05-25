"""
Sentiment analysis on IMDB using TF-IDF + scikit-learn.

Usage:
    python main.py                      # full run, Logistic Regression
    python main.py --clf svm            # use LinearSVC instead
    python main.py --max-train 5000     # quick smoke-test on 5k samples
    python main.py --predict "The film was absolutely brilliant!"
    python main.py --stats              # show statistics from the saved model
"""

import argparse  # lets the user pass in options when running the program, like --predict or --clf
import pickle  # lets me save the trained program to a file and load it back later
from pathlib import Path  # a neat way to refer to file locations on the computer

from src.data import load_imdb  # brings in the part of the project that gets and cleans the reviews
from src.model import build_pipeline  # brings in the part that sets up the learning brain
from src.evaluate import print_report, top_features  # brings in the part that shows the results

MODEL_PATH = Path("model.pkl")  # this is where the trained program gets saved on the computer


def train(args):
    X_train, y_train, X_test, y_test = load_imdb(  # go get all the reviews, cleaned and ready
        max_train=args.max_train,  # if the user set a limit on training reviews, pass that along
        max_test=args.max_test,    # if the user set a limit on test reviews, pass that along too
    )

    print(f"\nBuilding pipeline  (classifier={args.clf}) ...")  # tell the user which brain is about to be used
    pipeline = build_pipeline(args.clf)  # build the word-to-numbers step and learning brain, connected together

    print("Fitting ...")  # let the user know the learning has started
    pipeline.fit(X_train, y_train)  # show the program all the training reviews so it can learn from them

    print("Predicting on test set ...")  # let the user know testing has started
    y_pred = pipeline.predict(X_test)  # ask the program to guess positive or negative for every test review

    print_report(y_test, y_pred)  # compare the guesses to the real answers and show how well it did
    top_features(pipeline)  # show which words the program thinks are most positive and most negative

    MODEL_PATH.write_bytes(pickle.dumps(pipeline))  # save the trained program to a file so it can be used again later
    print(f"\nModel saved → {MODEL_PATH}")  # tell the user the file has been saved


def show_stats(max_test):
    if not MODEL_PATH.exists():  # check whether a saved model file is actually there
        raise FileNotFoundError("No saved model found. Run without --stats first.")  # if not, stop and tell the user
    print("Loading saved model...")  # let the user know the saved model is being loaded
    pipeline = pickle.loads(MODEL_PATH.read_bytes())  # load the saved model back from the file

    print("Loading test reviews...")  # let the user know reviews are being fetched
    _, _, X_test, y_test = load_imdb(max_test=max_test)  # grab only the test reviews, skip the training ones

    print("Running predictions...")  # let the user know it is working
    y_pred = pipeline.predict(X_test)  # ask the model to guess every test review

    print_report(y_test, y_pred)  # show the accuracy, precision, recall and confusion matrix
    top_features(pipeline)  # show the most positive and most negative words


def predict_single(text: str):
    if not MODEL_PATH.exists():  # check whether a saved program file is actually there
        raise FileNotFoundError("No saved model found. Run without --predict first.")  # if not, stop and tell the user
    pipeline = pickle.loads(MODEL_PATH.read_bytes())  # load the saved program back from the file
    label = pipeline.predict([text])[0]  # feed the review into the program and get back a 0 or 1
    label_names = {0: "NEGATIVE", 1: "POSITIVE"}  # turn that 0 or 1 into a word a person can read
    print(f"\nInput : {text!r}")  # show the review the user typed in
    print(f"Label : {label_names[label]}")  # show whether the program thinks it is positive or negative

    if hasattr(pipeline.named_steps["clf"], "predict_proba"):  # check if this brain type can also give a confidence percentage
        prob = pipeline.predict_proba([text])[0]  # ask for the percentage it's sure about each option
        print(f"Confidence : {max(prob):.2%}")  # show the highest confidence as a percentage


def main():
    parser = argparse.ArgumentParser(description="IMDB sentiment classifier")  # set up the reader that listens for options typed in the terminal
    parser.add_argument("--clf", choices=["lr", "svm"], default="lr",
                        help="Classifier: lr (Logistic Regression) or svm (LinearSVC)")  # lets the user pick which brain to use
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap training samples (e.g. 5000 for quick runs)")  # lets the user limit how many reviews to learn from
    parser.add_argument("--max-test", type=int, default=None,
                        help="Cap test samples")  # lets the user limit how many reviews to test on
    parser.add_argument("--predict", type=str, default=None,
                        help="Classify a single review string")  # lets the user type in one review to check
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics from the saved model without retraining")  # lets the user pull up stats any time
    args = parser.parse_args()  # actually read everything the user typed

    if args.predict:  # if the user typed a review to check...
        predict_single(args.predict)  # just check that one review and show the result
    elif args.stats:  # if the user asked for statistics...
        show_stats(args.max_test)  # load the saved model and show the stats
    else:
        train(args)  # otherwise go through the full process of learning and testing


if __name__ == "__main__":  # this makes sure the program only starts when you run this file directly
    main()  # kick everything off
