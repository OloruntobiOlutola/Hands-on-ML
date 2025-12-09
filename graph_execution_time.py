import matplotlib.pyplot as plt
import numpy as np

# Time in seconds for each model
time_by_model = {
    "Logistic Regression Sampled": 0.3,
    "LightGBM Sampled": 4.0,
    "CatBoost Sampled": 23.2,
    "Linear SVC Sampled": 0.2,
    "Random Forest Sampled": 87.9,
    "Gaussian Naive Bayes Sampled": 0.2,
    "Gradient Boosting Classifier Sampled": 308.6,
    "SVC Sampled": 241.2,

    "KNN Sampled": 1.9,
    "Multi-Layer Perceptron Sampled": 26.5,
    "XGBoost Sampled": 5.1,
    "Logistic Regression": 0.3,
    "CatBoost": 38.2,
    "LightGBM": 5.2,
    "Linear SVC": 1.7,
    "Random Forest": 1053.1,
    "Gaussian Naive Bayes": 0.5,
    "Gradient Boosting Classifier": 1649.2,
    "SVC": ,
}