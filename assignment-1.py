#!/usr/bin/env python3

from sklearn.datasets import load_iris
import numpy as np

# Load dataset
data = load_iris()
X = data.data  # features
y = data.target  # labels

print("\n--- LINEAR ALGEBRA (Matrix Ops) ---")
# Mean of each column (feature)
mean_vector = np.mean(X, axis=0)
print("Mean vector:", mean_vector)

# Covariance matrix
cov_matrix = np.cov(X.T)
print("Covariance matrix:\n", cov_matrix)

print("\n--- PROBABILITY ---")
# Probability of class 0 given petal length < 2
prob = np.mean(y[X[:, 2] < 2] == 0)
print("P(class 0 | petal length < 2):", prob)

print("\n--- OPTIMIZATION (Gradient Descent for Linear Regression) ---")
# Simplified gradient descent example
X_lr = X[:, :1]  # use first feature
X_lr = (X_lr - X_lr.mean()) / X_lr.std()  # normalize
X_lr = np.c_[np.ones(X_lr.shape[0]), X_lr]  # add bias

y_lr = y.astype(float)
weights = np.random.randn(2)
lr = 0.01

for _ in range(100):
    preds = X_lr @ weights
    error = preds - y_lr
    grad = X_lr.T @ error / len(X_lr)
    weights -= lr * grad

print("Weights after gradient descent:", weights)

print("\n--- CAPSTONE IDEA ---")
print("Capstone Idea: An AI tool to help students revise smarter by identifying weak topics based on performance data and recommending study plans.")
