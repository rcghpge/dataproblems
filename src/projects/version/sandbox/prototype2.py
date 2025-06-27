"""
@article{2019arXiv,
  author = {Saxton, Grefenstette, Hill, Kohli},
  title = {Analysing Mathematical Reasoning Abilities of Neural Models},
  year = {2019},
  journal = {arXiv:1904.01557}
}
"""

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics


# Load data
train = load_files('math/train-medium', encoding='utf-8')
test  = load_files('math/interpolate', encoding='utf-8')

# Map X to target y for binary classification
X = train
y = test


# Pipeline: vectorize + classify
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train/test split + training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=578)
pipe.fit(X_train, y_train)

# Evaluate
pred = pipe.predict(X_test)
#print(metrics.classification_report(y_test, pred, target_names=))






