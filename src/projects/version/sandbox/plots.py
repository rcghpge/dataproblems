from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Compute QA pair similarity
sims = np.array([cosine_similarity(q.reshape(1,-1), a.reshape(1,-1))[0,0]
                 for q, a in zip(Q_esa, A_esa)])
plt.hist(sims, bins=30)
plt.title("Question-Answer Semantic Similarity (Semantic Anylsis)")
plt.show()

