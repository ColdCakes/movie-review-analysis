# IMDB Sentiment Analysis

A machine learning project that reads a movie review and decides whether it is positive or negative. Built using Python and scikit-learn.

Achieves **90.58% accuracy** on 25,000 real IMDB reviews.

---

## How it works

1. Downloads 50,000 real IMDB movie reviews
2. Cleans the text by removing HTML, punctuation and converting to lowercase
3. Converts each review into numbers using TF-IDF
4. Trains a Logistic Regression model to learn which words mean positive and which mean negative
5. Tests it on 25,000 reviews it has never seen before

---

## How to run it

**Install the required tools (only needed once):**
```
pip install -r requirements.txt
```

**Full training run:**
```
python main.py
```

**Quick test on a smaller number of reviews:**
```
python main.py --max-train 2000 --max-test 500
```

**Test your own review:**
```
python main.py --predict "This was the best movie I have ever seen!"
```

**See statistics from the saved model:**
```
python main.py --stats
```

---

## Dataset

This project uses the Stanford IMDB Dataset — 50,000 real movie reviews labeled as positive or negative.

- Dataset source: https://ai.stanford.edu/~amaas/data/sentiment/
- Hosted on HuggingFace: https://huggingface.co/datasets/stanfordnlp/imdb

**Citation:**
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T.
               and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association
               for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
