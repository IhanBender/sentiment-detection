import os
import sys
import pandas as pd
import random
import sys
import os

root_dir = os.path.join(os.path.dirname(__file__), "../../")
raw_data = os.path.join(root_dir, "data/raw/")

reviews = pd.read_csv(os.path.join(raw_data, "olist_order_reviews_dataset.csv"),
                      usecols=["review_score", "review_comment_message"]).fillna("")
reviews = reviews[reviews.review_comment_message != ""]

separated = dict()
for score, comment in set(list(map(tuple, reviews.values))):
    try:
        separated[score].add(comment)
    except:
        separated[score] = {comment}

train, test = [], []
for key in separated:
    test_examples = random.sample(separated[key], k=int(len(separated[key]) * 0.2))
    train_examples = [x for x in separated[key] if x not in test_examples]
    test_examples = list(test_examples)

    for example in test_examples:
        test.append([key, example])
    for example in train_examples:
        train.append([key, example])

random.shuffle(train)
train = pd.DataFrame(train, columns=["score", "message"])
train.to_csv(os.path.join(root_dir, "data/interim/train.csv"))

random.shuffle(test)
test = pd.DataFrame(test, columns=["score", "message"])
test.to_csv(os.path.join(root_dir, "data/interim/test.csv"))