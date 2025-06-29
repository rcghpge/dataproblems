# EDA in Bash for Math + NLP


```bash
Semantic Analysis
- Uses external knowledge (e.g., Wikipedia) to map text to concept vectors.

- Captures semantic relatedness at a concept level, beyond simple word overlap

- Maps to known semantic concepts in a scientific discipline or subject matter

- Semantic analysis maps text into an interpretable concept-space based on real documents 

- Ideal for measuring semantic similarity or using high-level features.
```

# To save results:
# vectorizer
```python
import joblib
joblib.dump(concept_vec, "esa_vectorizer.joblib")
```

# concept matrix
```python
sp.save_npz("X_concepts.npz", X_concepts)
```
# vectors
```python
np.save("Q_esa.npy", Q_esa)
np.save("A_esa.npy", A_esa)

```

