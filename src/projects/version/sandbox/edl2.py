import pandas as pd

df = pd.read_csv("/qa.csv")  # CSV with 'question' & 'answer'

# Transform with same vocabulary
Q_tfidf = concept_vec.transform(df['question'])
A_tfidf = concept_vec.transform(df['answer'])

# Project into concept space
Q_sa = Q_tfidf.dot(X_concepts.T).A
A_sa = A_tfidf.dot(X_concepts.T).A

