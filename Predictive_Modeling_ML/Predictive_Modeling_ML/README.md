# Predictive Modeling Using Machine Learning

A complete supervised machine learning project that predicts whether a tumor is **malignant or benign** using the Breast Cancer dataset from scikit-learn.

## Features
- Data loading and exploration
- Train/test split
- Decision Tree and Random Forest models
- Accuracy, precision, recall and F1-score
- Confusion Matrix visualization
- ROC Curve visualization
- Model comparison

## Project Structure
```
Predictive_Modeling_ML/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── train.py
└── results/
    └── (generated after running the program)
```

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/train.py
```

The program saves:
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/model_comparison.csv`

## Algorithms Used
1. Decision Tree Classifier
2. Random Forest Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Expected Outcome
This project demonstrates the complete supervised learning workflow:
data preparation → model training → prediction → evaluation → visualization.

## GitHub
Upload this entire folder to a new GitHub repository named:
`predictive-modeling-ml`
