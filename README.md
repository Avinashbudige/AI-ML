# AI-ML POC — Fundamental Skills Demonstration

Purpose
- Create a set of reproducible, well-documented notebooks and small projects that demonstrate practical, interview-level fundamentals across:
  - Numpy, Pandas, Python LLD, Statistics
  - Supervised learning, Ensemble techniques, Unsupervised learning, Feature engineering, Model selection & tuning
  - Intro to neural networks & deep learning, Computer vision, NLP, GANs, Reinforcement learning

Goals
- Each topic gets 1 (or more) focused notebook with:
  - clear objective
  - dataset & preprocessing
  - concise EDA
  - modeling baseline(s)
  - evaluation & interpretation
  - short write-up of conclusions and next steps
- Provide a rubric and interview-style questions & coding tasks so the POC can serve as verifiable proof of knowledge.

Repository structure (recommended)
- README.md (this file)
- project_plan.md — mapping topics to notebooks, datasets and timeline
- rubric.md — scoring rubric, interview questions and coding tasks
- requirements.txt — reproducible environment
- notebooks/
  - 00_template.py — notebook template
  - 01_numpy_pandas_basics.ipynb
  - 02_statistics_exploratory_analysis.ipynb
  - 03_supervised_baseline_models.ipynb
  - 04_ensemble_methods.ipynb
  - 05_feature_engineering_and_selection.ipynb
  - 06_model_selection_and_hyperparam_tuning.ipynb
  - 07_unsupervised_clustering_dimensionality_reduction.ipynb
  - 08_intro_neural_networks_tf_pytorch.ipynb
  - 09_computer_vision_classification_object_detection.ipynb
  - 10_nlp_text_classification_embedding_seq_models.ipynb
  - 11_gan_basic_generative_models.ipynb
  - 12_reinforcement_learning_basic_envs.ipynb
- data/ (gitignored; pointers to download scripts)
- outputs/ (models, figures, saved artifacts; gitignored)

How to run
1. Create an environment:
   - python >= 3.8
   - pip install -r requirements.txt
2. Open the notebooks in Jupyter / JupyterLab
3. Use data-download cells or provided scripts to fetch datasets (instructions in each notebook)

Datasets (suggestions)
- Tabular / ML fundamentals: UCI datasets (e.g., Wine, Adult), Kaggle Titanic
- Classification (vision): MNIST, Fashion-MNIST, CIFAR-10
- Object detection / advanced CV: subset of COCO or Pascal VOC (for advanced POC)
- NLP: IMDB sentiment, AG News, small subset of Wikipedia or SST
- GAN demo: MNIST or CelebA (small subset)
- RL: OpenAI Gym CartPole and MountainCar

Deliverables per notebook
- Notebook with clear sections: objective, imports, data, EDA, baseline, improvements, evaluation, conclusions
- One saved artifact (trained model or notebook outputs) and README snippet summarizing results

License
- Add an appropriate license (MIT recommended for POCs)
