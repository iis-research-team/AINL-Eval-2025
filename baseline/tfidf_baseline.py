from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_PATH = Path(__file__).parent.parent.parent

categories = ['llama-3.3-70b', 'gemma-2-27b', 'gpt-4-turbo', 'human']


# create mapping
def labels2id(labels: List[str]) -> List[int]:
    ids = []
    for label in labels:
        ids.append(categories.index(label))
    return ids

# load data
train_data_fpath = PROJECT_PATH / 'data' / 'train.csv'
train_data = pd.read_csv(train_data_fpath)

dev_data_fpath = PROJECT_PATH / 'data' / 'dev.csv'
dev_data = pd.read_csv(dev_data_fpath)

train_data = train_data.replace(np.nan, '')

train_texts = list(train_data['text'])
train_labels = list(train_data['label'])
train_ids = labels2id(train_labels)

dev_texts = list(dev_data['text'])
dev_ids = list(dev_data['id'])

# create vectorizer
print('Running vectorizer...')
vectorizer = TfidfVectorizer(
     sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)

feature_names = vectorizer.get_feature_names_out()

# create classifier
clf = LogisticRegression(C=5, max_iter=1000)
clf.fit(X_train, train_ids)

preds = []

# predict texts with the classifier
for sample in X_dev:
    prob = clf.predict_proba(sample)[0]
    max_prob = max(prob)
    max_label = np.argmax(prob)

    if max_prob > 0.6:
        preds.append(categories[max_label])
    else:
        preds.append('unknown')

# save submission
submission_df = pd.DataFrame({'id': ids, 'label': preds})
submission_df.to_csv('submission.csv', index=False)
