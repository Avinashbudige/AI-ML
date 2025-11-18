# Project Plan & Mapping (POC for interview-proof)

Overview
- Build one focused notebook per topic. Prefer small, reproducible datasets to keep notebooks fast and reviewable.
- Each notebook must be self-contained: a reviewer should open it and reproduce results within ~10 minutes (or with a noted short download step).

Mapping: topic -> notebook objective -> suggested dataset -> deliverable
- Numpy & Pandas (01_numpy_pandas_basics.ipynb)
  - Objective: demonstrate array ops, broadcasting, vectorization, groupby, joins, pivoting, missing data handling
  - Dataset: synthetic & small CSV (Titanic subset)
  - Deliverable: short script showing timing of vectorized vs loop approaches + cleaned dataframe

- Statistics & EDA (02_statistics_exploratory_analysis.ipynb)
  - Objective: descriptive stats, distributions, central tendency, hypothesis testing, confidence intervals
  - Dataset: UCI Wine or Iris
  - Deliverable: report with tests (t-test, chi-sq) and interpretations

- Python LLD (design patterns, modular code) (integrated across notebooks)
  - Objective: show small module or class for data pipeline and a unit test
  - Deliverable: a python module under notebooks/utils.py and a pytest unit test

- Supervised Learning Baseline (03_supervised_baseline_models.ipynb)
  - Objective: linear/logistic regression, decision tree baseline
  - Dataset: UCI Adult, or Kaggle Titanic
  - Deliverable: baseline metrics and short model card

- Ensemble Techniques (04_ensemble_methods.ipynb)
  - Objective: bagging, random forest, boosting (XGBoost/LightGBM)
  - Dataset: structured classification/regression (Adult/Wine)
  - Deliverable: comparison table (accuracy/AUC/MAE) + feature importances

- Feature Engineering & Selection (05_feature_engineering_and_selection.ipynb)
  - Objective: handling categorical variables, scaling, interaction terms, feature selection (L1, recursive)
  - Dataset: any above
  - Deliverable: before/after performance + selected features list

- Model Selection & Tuning (06_model_selection_and_hyperparam_tuning.ipynb)
  - Objective: cross-validation, grid/random search, Bayesian (optional), nested CV
  - Dataset: same as supervised
  - Deliverable: tuned model + explanation of CV strategy

- Unsupervised Learning (07_unsupervised_clustering_dimensionality_reduction.ipynb)
  - Objective: KMeans, hierarchical, DBSCAN; PCA/TSNE/UMAP visualizations
  - Dataset: MNIST subset (for visual clusters) or Iris
  - Deliverable: cluster metrics and visual plots

- Intro to Neural Networks & Deep Learning (08_intro_neural_networks_tf_pytorch.ipynb)
  - Objective: build & train a small MLP in PyTorch or TF; explain backprop, loss, optimizer
  - Dataset: MNIST or Fashion-MNIST
  - Deliverable: training curve + saved model + short explanation

- Computer Vision (09_computer_vision_classification_object_detection.ipynb)
  - Objective: CNN for classification; demo transfer learning (ResNet, torchvision)
  - Dataset: CIFAR-10 or subset of ImageNet / Cats vs Dogs
  - Deliverable: final accuracy and confusion matrix, sample CAM/Grad-CAM explanation

- NLP (10_nlp_text_classification_embedding_seq_models.ipynb)
  - Objective: classic tokenization + TF-IDF + linear classifier; demo embeddings (word2vec/fastText) and an LSTM or Transformer small demo
  - Dataset: IMDB or AG News
  - Deliverable: comparison of models + example inference pipeline

- GAN (11_gan_basic_generative_models.ipynb)
  - Objective: simple DCGAN on MNIST to show generator/discriminator and training stability tips
  - Dataset: MNIST
  - Deliverable: saved generator outputs and short commentary about mode collapse and evaluation (FID not required)

- Reinforcement Learning (12_reinforcement_learning_basic_envs.ipynb)
  - Objective: implement/visualize policy for CartPole using a simple DQN or policy gradient (stable-baselines3 or minimal custom)
  - Dataset: OpenAI Gym envs
  - Deliverable: training curve and rendered episode GIF or video

Timeline suggestions
- 4 weeks (intensive): 2-3 notebooks per week, focus on breadth with a few deep demos (NN, CV, NLP)
- 8 weeks (moderate): allow more polish, tests, README, model cards

Notes
- Keep all code modular and re-usable.
- Keep notebooks short and descriptive; move heavy code into notebooks/utils.py or scripts.
