import os

files = ["sa_vectorizer.joblib", "X_concepts.npz", "Q_sa.npy", "A_sa.npy"]
for f in files:
    print(f, "\nChecking vector embeddings" , os.path.exists(f))

