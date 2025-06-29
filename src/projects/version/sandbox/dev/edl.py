from sklearn.feature_extraction.text import TfidfVectorizer
import glob

# Load data (e.g., for research and exploratory data analysis)
docs = []
for path in glob.glob("math/train-medium/*.txt"):
    with open(path) as f:
        docs.append(f.read())

concept_vec = TfidfVectorizer(stop_words='english', max_features=10000)
X_concepts = concept_vec.fit_transform(docs)  # shape: (n_concepts × vocabulary)

