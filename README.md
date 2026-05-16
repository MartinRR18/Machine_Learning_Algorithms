# Machine Learning Algorithms and Neural Networks

This repository contains a collection of projects related to machine learning algorithms, with a focus on quantitative finance, time series analysis, and deep learning applications. The projects range from foundational implementations of classical algorithms to advanced deep learning models and Bayesian networks.

## Table of Contents

1.  [Shallow and Deep Neural Networks for MNIST-1D Classification](#1-shallow-and-deep-neural-networks-for-mnist-1d-classification)
2.  [Hidden Naive Bayes Classifier](#2-hidden-naive-bayes-classifier)
3.  [Bayesian Networks: Structure Learning with Genetic Algorithms](#3-bayesian-networks-structure-learning-with-genetic-algorithms)
4.  [CAIM Discretization Algorithm](#4-caim-discretization-algorithm)
5.  [Anomaly Detection using Autoencoders and Deep Neural Networks](#5-anomaly-detection-using-autoencoders-and-deep-neural-networks)
6.  [Mixture Models for Data Fitting](#6-mixture-models-for-data-fitting)
7.  [General Dependencies](#7-general-dependencies)

---

### 1. Shallow and Deep Neural Networks for MNIST-1D Classification

*   **File:** `DL1_SD-NNs-MNIST-1D.ipynb`
*   **Description:** This Jupyter notebook explores the implementation and training of Shallow and Deep Neural Networks for classifying the 1D version of the MNIST dataset. It delves into architectural differences, activation functions, optimizers, and techniques to mitigate overfitting.
*   **Key Concepts:**
    *   MNIST-1D dataset characteristics and generation.
    *   Feed-forward neural network architectures.
    *   Activation functions: ReLU, Softmax, PReLU.
    *   Optimizers: Adam, Stochastic Gradient Descent (SGD) with Momentum, RMSprop.
    *   Loss function: Categorical Cross-Entropy.
    *   Regularization techniques: Early Stopping based on loss and accuracy ratios.
    *   Data preprocessing: NumPy array to PyTorch tensors, train-validation split.

### 2. Hidden Naive Bayes Classifier

*   **File:** `HiddenNaiveBayes.py`
*   **Description:** This Python script provides an implementation of the Hidden Naive Bayes (HNB) algorithm, based on the paper "A novel Bayes Model: Hidden Naive Bayes" by Jiang, Zhang, and Cai (2008). HNB extends the traditional Naive Bayes classifier by introducing "hidden parents" to account for attribute dependencies.
*   **Key Concepts:**
    *   Prior class probabilities.
    *   Conditional Mutual Information for calculating feature weights.
    *   Conditional probabilities given hidden parents.
    *   Prediction mechanism for new instances.
    *   Dependencies: `numpy`, `pandas`, `sklearn.model_selection`, `sklearn.metrics`, `tqdm`.

### 3. Bayesian Networks: Structure Learning with Genetic Algorithms

*   **Files:** `BayesNetwork.ipynb`, `BayesNetwork_Restrictions.ipynb`
*   **Description:** These notebooks focus on learning the structure of Bayesian Networks (represented as Directed Acyclic Graphs - DAGs) from data. It implements a search-and-scoring approach using a genetic algorithm with the BIC (Bayesian Information Criterion) score. The `BayesNetwork_Restrictions.ipynb` file explores methods for calculating dependencies and independencies between nodes.
*   **Key Concepts:**
    *   Representing Bayesian Networks as Adjacency Matrices (DAGs).
    *   BIC (Bayesian Information Criterion) as a scoring function for network structures.
    *   Genetic Algorithm components:
        *   Population initialization with random DAGs.
        *   Evaluation using the BIC score.
        *   Parent selection (roulette wheel selection).
        *   Crossover operations with random binary masks.
        *   Mutation operations (add, delete, flip edges).
        *   Acyclicity validation and repair mechanisms for DAGs.
    *   Mutual Information and Conditional Mutual Information for dependency analysis.
    *   Chi-square test for independence.
    *   Dependencies: `numpy`, `pandas`, `networkx`, `matplotlib`, `scipy`.

### 4. CAIM Discretization Algorithm

*   **File:** `CAIM.py`
*   **Description:** This script implements the CAIM (Class-Attribute Interdependence Maximization) algorithm, a supervised discretization method. CAIM aims to find optimal cut-points for continuous features by maximizing the class-attribute interdependence, ensuring high class purity within the resulting bins.
*   **Key Concepts:**
    *   Supervised discretization.
    *   Class-attribute interdependence.
    *   Algorithm steps: Candidate split points, frequency matrix construction, CAIM score calculation.
    *   Dependencies: `numpy`, `pandas`, `sklearn.preprocessing.LabelEncoder`.

### 5. Anomaly Detection using Autoencoders and Deep Neural Networks

*   **File:** `AnomalyDetection_Autoencoders_DNNs.ipynb`
*   **Description:** This project demonstrates a pipeline for anomaly detection using Autoencoders, followed by classification with Deep Neural Networks (DNNs). It uses the Optdigits dataset to illustrate how autoencoders can identify anomalous data points based on reconstruction error, and how cleaning the dataset can improve classification performance.
*   **Key Concepts:**
    *   Autoencoder architecture: Encoder and Decoder, PReLU activation.
    *   Loss function for autoencoders: Mean Squared Error (MSE).
    *   Optimizer for autoencoders: Adam.
    *   Reconstruction error as an anomaly score.
    *   Anomaly thresholding and data purging.
    *   Deep Neural Network for classification with PReLU activations.
    *   Data preprocessing: Standardization, tensor conversion, DataLoader setup.
    *   Dependencies: `torch`, `torch.nn`, `torch.optim`, `sklearn.preprocessing`, `sklearn.model_selection`, `matplotlib`, `ucimlrepo`.

### 6. Mixture Models for Data Fitting

*   **File:** `MixtureModels.ipynb`
*   **Description:** This notebook explores the application of mixture models to fit various data distributions. It covers both Gaussian Mixture Models (GMM) using scikit-learn and a custom implementation of the Expectation-Maximization (EM) algorithm for a mixture of Exponential and Power Law distributions.
*   **Key Concepts:**
    *   Gaussian Mixture Models (GMM): `n_components`, `weights_`, `means_`, `covariances_`.
    *   Expectation-Maximization (EM) algorithm: E-step (responsibility calculation), M-step (parameter update).
    *   Custom probability distribution functions (PDFs) using `scipy.stats.rv_continuous`.
    *   Fitting data to a mixture of Exponential and Power Law distributions.
    *   Dependencies: `numpy`, `scipy.stats`, `scipy.special`, `sklearn.mixture`, `matplotlib`, `seaborn`.

---

### 7. General Dependencies

Most projects in this repository rely on the following Python libraries:

*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `torch`
*   `matplotlib`
*   `networkx`
*   `scipy`
*   `tqdm` (for progress bars)
*   `python-binance` (for Binance_Bot project, not detailed above but often used in the workspace)
*   `yfinance` (for stock data fetching, not detailed above but often used in the workspace)

It is recommended to use a virtual environment to manage these dependencies.
