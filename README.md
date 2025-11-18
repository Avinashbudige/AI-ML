# AI/ML/DL Comprehensive Interview Preparation Curriculum

A complete, hands-on curriculum covering fundamental to advanced topics in Artificial Intelligence, Machine Learning, and Deep Learning, designed specifically for job interview preparation.

## 📚 Curriculum Overview

This repository contains 12 comprehensive Jupyter notebooks covering all essential topics that interviewers look for in AI/ML/DL roles:

### **Foundational Topics**

1. **[01_numpy_pandas_basics.ipynb](01_numpy_pandas_basics.ipynb)**
   - NumPy arrays and operations
   - Pandas DataFrames and Series
   - Data manipulation and cleaning
   - Memory-efficient operations
   - Interview Q&A included

2. **[02_statistics_exploratory_analysis.ipynb](02_statistics_exploratory_analysis.ipynb)**
   - Descriptive statistics
   - Probability distributions
   - Hypothesis testing
   - Correlation and covariance
   - Outlier detection
   - EDA best practices

3. **[03_supervised_baseline_models.ipynb](03_supervised_baseline_models.ipynb)**
   - Linear & Logistic Regression
   - Decision Trees
   - K-Nearest Neighbors
   - Naive Bayes
   - Support Vector Machines
   - Model evaluation metrics

### **Intermediate Topics**

4. **[04_ensemble_methods.ipynb](04_ensemble_methods.ipynb)**
   - Bagging & Random Forest
   - Boosting (AdaBoost, Gradient Boosting)
   - XGBoost & LightGBM
   - Voting & Stacking

5. **[05_feature_engineering_and_selection.ipynb](05_feature_engineering_and_selection.ipynb)**
   - Feature scaling & normalization
   - Encoding categorical variables
   - Feature creation & extraction
   - Selection methods (RFE, SelectKBest)
   - Handling imbalanced data

6. **[06_model_selection_and_hyperparameter_tuning.ipynb](06_model_selection_and_hyperparameter_tuning.ipynb)**
   - Grid Search & Random Search
   - Cross-validation strategies
   - Learning curves
   - Bias-variance tradeoff
   - Model selection strategies

7. **[07_unsupervised_learning___clustering_and_dimensionality_reduction.ipynb](07_unsupervised_learning___clustering_and_dimensionality_reduction.ipynb)**
   - K-Means & Hierarchical Clustering
   - DBSCAN
   - PCA & t-SNE
   - UMAP
   - Cluster evaluation

### **Deep Learning Topics**

8. **[08_introduction_to_neural_networks_with_tensorflow_and_pytorch.ipynb](08_introduction_to_neural_networks_with_tensorflow_and_pytorch.ipynb)**
   - Neural network fundamentals
   - Activation functions
   - Backpropagation
   - TensorFlow/Keras implementation
   - PyTorch basics
   - Training & optimization

9. **[09_computer_vision___classification_and_object_detection.ipynb](09_computer_vision___classification_and_object_detection.ipynb)**
   - CNN architectures
   - Image classification
   - Transfer learning (ResNet, VGG, etc.)
   - Object detection (YOLO, R-CNN)
   - Data augmentation

10. **[10_nlp___text_classification_embeddings_and_sequence_models.ipynb](10_nlp___text_classification_embeddings_and_sequence_models.ipynb)**
    - Text preprocessing
    - Word embeddings (Word2Vec, GloVe)
    - RNN & LSTM
    - Text classification
    - Attention mechanisms
    - Transformer basics

### **Advanced Topics**

11. **[11_gans_and_basic_generative_models.ipynb](11_gans_and_basic_generative_models.ipynb)**
    - GAN architecture
    - Generator & Discriminator
    - Training GANs
    - DCGAN
    - Variational Autoencoders (VAE)

12. **[12_reinforcement_learning___basic_environments.ipynb](12_reinforcement_learning___basic_environments.ipynb)**
    - RL fundamentals
    - Q-Learning
    - Deep Q-Networks (DQN)
    - Policy gradients
    - OpenAI Gym environments

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- 8GB RAM minimum (16GB recommended for deep learning notebooks)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Avinashbudige/AI-ML.git
cd AI-ML
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter:**
```bash
jupyter notebook
# or
jupyter lab
```

5. **Start with notebook 01** and progress sequentially through the curriculum.

## 📋 What Makes This Curriculum Interview-Ready?

### ✅ Comprehensive Coverage
- All fundamental ML/DL algorithms
- Real-world implementation examples
- Best practices and common pitfalls

### ✅ Interview-Focused Content
- Each notebook includes common interview questions
- Conceptual explanations with practical code
- Trade-offs and when to use each technique

### ✅ Hands-On Learning
- Working code examples
- Practice exercises
- Real datasets and scenarios

### ✅ Industry-Standard Tools
- scikit-learn for classical ML
- TensorFlow & PyTorch for deep learning
- XGBoost, LightGBM for production ML
- Modern libraries (transformers, stable-baselines3)

## 🎯 Learning Path

### For Beginners
Start from notebook 01 and follow sequentially through 07. This covers all foundational ML concepts.

### For Intermediate Learners
If comfortable with basics, start from notebook 04 (Ensemble Methods) and continue through deep learning topics.

### For Interview Preparation
1. Review relevant notebooks for your target role
2. Complete practice exercises
3. Study interview questions in each notebook
4. Implement algorithms from scratch for deeper understanding

## 📊 Project Structure

```
AI-ML/
├── 01_numpy_pandas_basics.ipynb
├── 02_statistics_exploratory_analysis.ipynb
├── 03_supervised_baseline_models.ipynb
├── 04_ensemble_methods.ipynb
├── 05_feature_engineering_and_selection.ipynb
├── 06_model_selection_and_hyperparameter_tuning.ipynb
├── 07_unsupervised_learning___clustering_and_dimensionality_reduction.ipynb
├── 08_introduction_to_neural_networks_with_tensorflow_and_pytorch.ipynb
├── 09_computer_vision___classification_and_object_detection.ipynb
├── 10_nlp___text_classification_embeddings_and_sequence_models.ipynb
├── 11_gans_and_basic_generative_models.ipynb
├── 12_reinforcement_learning___basic_environments.ipynb
├── requirements.txt
└── README.md
```

## 🔧 Key Libraries Used

- **Data Processing:** NumPy, Pandas, SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **NLP:** NLTK, spaCy, Transformers
- **Computer Vision:** OpenCV, Pillow
- **Reinforcement Learning:** Gym, Stable-Baselines3

## 💡 Tips for Success

1. **Practice coding from scratch** - Don't just run the notebooks, implement algorithms yourself
2. **Understand the math** - Know the theory behind each algorithm
3. **Compare approaches** - Understand when to use each technique
4. **Work on projects** - Apply knowledge to real datasets
5. **Review regularly** - Revisit concepts before interviews

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Add more examples or exercises
- Fix typos or improve documentation

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

**Avinash Budige**
- GitHub: [@Avinashbudige](https://github.com/Avinashbudige)

## 🙏 Acknowledgments

This curriculum is designed based on:
- Common AI/ML/DL interview questions
- Industry best practices
- Academic research and standard textbooks
- Real-world project requirements

---

**Good luck with your AI/ML/DL interview preparation! 🚀**

*Remember: Understanding concepts deeply is more important than memorizing code. Focus on the "why" and "when" along with the "how".*
