# Grading Rubric & Interview Proof Checklist

Scoring (suggested)
- 0–2: Missing or incorrect
- 3–4: Basic understanding, small issues
- 5: Clear, correct, well-documented

Rubric categories (per notebook)
- Correctness (0–5)
- Reproducibility (0–5) — clear instructions, environment, seeds
- Explanation & Interpretation (0–5) — why choices were made, limitations
- Code quality & modularity (0–5) — functions, small classes, tests
- Results & Visualization (0–5) — clarity of plots, metrics and comparisons

Example total: 25 per notebook; pick 3–4 flagship notebooks (NN, CV, NLP, Supervised) for a concise portfolio.

Interview-style questions per topic (examples)
- Numpy/Pandas: vectorize a column operation (no loops); implement groupby + custom aggregation; differences between .loc and .iloc
- Statistics: interpret p-values and confidence intervals; when to use t-test vs chi-square; bias vs variance
- Python LLD: design a small DataPipeline class with transform/fit/persist methods; how to unit test it
- Supervised: explain model bias and variance tradeoff; how would you handle class imbalance
- Ensemble: difference between bagging and boosting; when to use stacking
- Featureing: encode cyclical features (time of day); how to use target encoding and avoid leakage
- Model selection & tuning: describe nested CV; when to use random search vs grid search
- NN & DL: explain backprop and vanishing gradients; effect of batch normalization, dropout
- CV: transfer learning vs training from scratch; explain Grad-CAM
- NLP: pros/cons of TF-IDF vs embeddings; fine-tuning vs feature-based for transformers
- GAN: describe generator/discriminator objectives; signs of mode collapse
- RL: explain exploration vs exploitation; difference between policy gradient and Q-learning

Coding tasks (short)
- Implement a data cleaning pipeline that imputes missing values and encodes categorical vars; show before/after classifier AUC change.
- Build a small CNN that reaches >75% on Fashion-MNIST within a resource-limited setting (few epochs).
- Implement KMeans from scratch (distance, update centroids) and compare with sklearn KMeans on small dataset.

Proof artifacts to provide
- Notebooks (clean, executed) + short README per notebook
- Trained model checkpoint(s) (small)
- One or two short videos/GIFs (for RL or CV demo) optional
- Unit tests for at least the utilities module
