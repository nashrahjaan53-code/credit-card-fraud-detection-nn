# Credit Card Fraud Detection with Cost-Optimized Neural Networks

[![Python - Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework - PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Imbalance Technique - SMOTE](https://img.shields.io/badge/Imbalance--Handling-SMOTE-orange)](https://github.com/scikit-learn-contrib/imbalanced-learn)
[![Optimization Metric - AUPRC](https://img.shields.io/badge/Optimization--Metric-AUPRC%20%26%20F2--Score-brightgreen)](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
[![License - MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a production-ready, leak-free machine learning pipeline designed to detect fraudulent credit card transactions. Instead of using standard classification thresholds, this project applies a custom financial cost optimization matrix to minimize operational losses for enterprise banking systems.

---

## Machine Learning Pipeline Architecture

The end-to-end framework decouples data engineering, deep learning training, and business-metric decision systems into separate phases:

```text
[Raw creditcard.csv] ──► [Strict 80/20 Train-Test Split]
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
             [Training Data]          [Test Data (Holdout)]
                     │                       │
           [Fit/Apply Scaler]         [Apply Scaler Only]
                     │                       │
           [Apply SMOTE (10%)]               │
                     │                       │
                     ▼                       ▼
            [Untouched Reality]     [Untouched Reality]
                     │                       │
                     ▼                       ▼
            [Train Classifier]       [Extract Probabilities]
                     │                       │
                     └───────────┬───────────┘
                                 ▼
                    [Custom Financial Cost Matrix]
                                 │
                                 ▼
                     [Optimal Threshold Found]
fraud-detection-nn/
│
├── data/
│   └── .gitkeep                 # Preserves data directory structure in version control
│
├── notebooks/
│   ├── 1_eda_and_smote.ipynb    # Data exploration and imbalance analysis
│   └── 2_nn_training_tuning.ipynb # Interactive evaluation and visualization
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Leakage-free preprocessing, scaling, and sampling
│   ├── model.py                 # PyTorch Multi-Layer Perceptron architecture
│   └── engine.py                # Training loops and financial threshold tuning
│
├── .gitignore                   # Excludes raw datasets and local cache files
├── main.py                      # Master pipeline orchestrator
└── requirements.txt             # Project dependencies

Key Engineering Challenges Addressed
1. Strict Class Imbalance (0.17% Minority Class)
The dataset consists of 284,807 transactions, where only 492 are fraudulent. Relying on basic classification accuracy yields a deceptive 99.83% score by simply predicting "legitimate" across all cases. This pipeline prioritizes Area Under the Precision-Recall Curve (AUPRC) to isolate performance on the minority class.

2. Eliminating Data Leakage
A common mistake in fraud modeling is applying oversampling techniques to the entire dataset before splitting. This pipeline strictly partitions the data into Train/Test subsets first. Standard feature scaling and SMOTE oversampling are applied exclusively to the training set. The validation and test sets remain completely untouched and imbalanced, reflecting true production environments.

3. Business Cost Optimization Matrix
In commercial banking, the cost of a False Negative (missing actual fraud) is significantly higher than a False Positive (generating a false alarm). This project incorporates a custom business utility function to sweep classification thresholds and find the exact balance point minimizing overall financial impact:

Metric Type	Operational Outcome	Assigned Cost	Business Rationale
False Negative (FN)	Model misses fraud; customer account compromised.	$500	Covers bank reimbursement costs and compliance fees.
False Positive (FP)	Model flags legitimate use; user transaction frozen.	$15	Covers operational cost of automated text or agent check.

Model Deep Learning Blueprint
The classifier is built as a highly stable PyTorch nn.Module optimized for tabular embeddings:

Plaintext


[29 Scaled Features] ──► [Linear 64] ──► [Batch Norm] ──► [ReLU] ──► [Dropout 30%] ─┐
                                                                                   │
[1 Output Logit]     ◄── [Linear 32] ◄── [Batch Norm] ◄── [ReLU] ◄── [Dropout 20%] ─┘
Batch Normalization (1D): Stabilizes internal covariate shift across training epochs.

Dropout Layering: Positioned after activation layers (30% and 20%) to mitigate overfitting on synthetic data points.

Logit-Based Outputs: Designed to work directly with BCEWithLogitsLoss for maximum numerical stability during backward gradient updates.
Getting Started
Prerequisites
Clone this repository and ensure Python 3.10+ is installed locally. Install the pinned dependencies using pip:

Bash


pip install -r requirements.txt

Dataset Setup
Download the raw transaction dataset from the Kaggle Credit Card Fraud Detection Page.

Extract the archive and place the creditcard.csv file directly into the data/ directory.

Execution
Run the master orchestrator script to trigger the full end-to-end data processing, model training, and threshold optimization pipeline:

Bash


python main.py