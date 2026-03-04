# Credit Card Fraud Detection

## Overview

Credit card fraud is one of the most critical problems faced by financial institutions today. Fraudulent transactions cause significant financial losses and reduce customer trust.

This project implements a **Machine Learning based Credit Card Fraud Detection System** that analyzes transaction data and identifies suspicious activities. The model learns patterns from historical transactions and classifies them as **fraudulent or legitimate**.

The goal of this project is to develop a scalable system that can assist financial institutions in detecting fraud in real-time.

---

## Problem Statement

Credit card fraud detection is challenging due to:

* Highly **imbalanced datasets**
* **Evolving fraud patterns**
* Need for **real-time detection**
* Large-scale transaction data

This project aims to build an intelligent system that can detect fraud with high accuracy while minimizing false positives.

---

## Objectives

* Detect fraudulent credit card transactions
* Train machine learning models on transaction data
* Handle class imbalance in the dataset
* Evaluate model performance using appropriate metrics
* Build a pipeline for fraud detection

---

## Dataset

The dataset contains anonymized credit card transactions with the following characteristics:

* Transactions made by European cardholders
* Highly imbalanced dataset
* Fraud transactions represent a very small percentage

Typical features include:

* Transaction time
* Transaction amount
* PCA-transformed features (V1 – V28)
* Transaction class (Fraud / Non-Fraud)

Class Labels:

* **0 → Legitimate Transaction**
* **1 → Fraudulent Transaction**

---

## System Architecture

```
Transaction Data
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Machine Learning Model
       │
       ▼
Fraud Prediction
       │
       ▼
Performance Evaluation
```

---

## Technologies Used

Programming Language

* Python

Libraries

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

Tools

* Jupyter Notebook
* Git
* GitHub

---

## Project Structure

```
Credit-Card-Fraud-Detection
│
├── data
│   └── creditcard.csv
│
├── notebooks
│   └── fraud_detection.ipynb
│
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models
│   └── trained_model.pkl
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Scaling
5. Handling Class Imbalance
6. Model Training
7. Model Evaluation
8. Fraud Prediction

---

## Machine Learning Models

The following algorithms can be used:

* Logistic Regression
* Random Forest
* Decision Tree
* Support Vector Machine
* Gradient Boosting

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Installation

Clone the repository:

```
git clone https://github.com/Adarshthakur-850/Credit-Card-Fraud-Detection.git
```

Navigate to project folder:

```
cd Credit-Card-Fraud-Detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Run the main script:

```
python main.py
```

Or open the notebook:

```
jupyter notebook
```

---

## Results

The trained model successfully detects fraudulent transactions by learning patterns in transaction behavior.

Key evaluation metrics include:

* High recall for fraud detection
* Balanced precision to reduce false alarms
* Improved fraud detection accuracy

---

## Future Improvements

* Real-time fraud detection system
* Deep learning models
* Integration with banking transaction APIs
* Deployment using cloud services
* MLOps pipeline for automated retraining

---

## Author

Adarsh Thakur

Machine Learning Engineer
Lovely Professional University

GitHub Profile:
[https://github.com/Adarshthakur-850](https://github.com/Adarshthakur-850)
