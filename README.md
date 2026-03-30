# PyTorch Lab

A collection of hands-on PyTorch exercises exploring core machine learning concepts from scratch.

## Exercises

### Linear Regression
Fits a simple `y = 2x + 1` relationship using gradient descent. Demonstrates tensor operations, MSE loss, and the basic training loop (forward pass, backpropagation, weight update).

### Logistic Regression
Binary classification on a 2D dataset with two clusters. Uses a sigmoid activation and BCE loss to learn a decision boundary separating the classes.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch
```

## Usage

```bash
python linear_regression.py
python logistic_regression.py
```
