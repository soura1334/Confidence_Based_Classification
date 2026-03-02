# Confidence-Based Classification with Reject Option

This project implements a confidence-aware classification system using Logistic Regression.
Unlike traditional classifiers that always produce a prediction, this model introduces a reject option — it abstains from making predictions when confidence is low.

The goal is to improve decision reliability rather than focusing only on accuracy.

# Problem Statement

Traditional classification models always force a prediction, even when the model is uncertain. This can lead to unreliable decisions in sensitive domains such as academic evaluation.

This project proposes a Confidence-Based Classification System with Reject Option, where predictions are made only if the probability exceeds a predefined confidence threshold. Otherwise, the prediction is rejected for manual review.

# Dataset Used

Dataset: Student Performance Dataset
Source: UCI Machine Learning Repository

File used:

student-mat.csv

Target Variable:

Final Grade (G3) converted to:

- 1 → Pass (G3 ≥ 10)

- 0 → Fail (G3 < 10)

Selected Features:

- studytime

- failures

- absences

- G1

- G2

# How the Model Works

Logistic Regression is trained on student performance data.

The model outputs probability of passing.

A confidence threshold (e.g., 0.7) is defined.

Decision rule:

- If probability ≥ threshold → PASS

- If probability ≤ (1 − threshold) → FAIL

- Otherwise → REJECT (low confidence)

This creates a trade-off between:

- Prediction coverage

- Prediction reliability

# Evaluation Metrics

The system evaluates performance using:

- Baseline Accuracy (no rejection)

- Accuracy on confident predictions

- Rejection Rate

This shifts the focus from "How accurate is the model?"
to
"How trustworthy is the model?"

# Sample Output

Example:

Model Performance
Accuracy (confident predictions): 0.9612
Rejection Rate: 0.1345

This means:

- 86% of cases received confident predictions

- Out of those, ~96% were correct

# Installation
1. Create Virtual Environment (Windows)
```
python -m venv venv
venv\Scripts\activate
```
2. Install Dependencies
```
pip install pandas numpy scikit-learn matplotlib jupyter
```
3. Run Notebook
```
jupyter notebook
```
# Project Structure
```text
confidence-based-classification/
│
├── Confidence_Based_Classification_Human_Written.ipynb
├── confidence_based_classification_modular.py
├── student-mat.csv
└── README.md
```

# References

- UCI Machine Learning Repository – Student Performance Dataset

- El-Yaniv, R. (2010). On the Foundations of Selective Classification

- Scikit-learn Documentation
