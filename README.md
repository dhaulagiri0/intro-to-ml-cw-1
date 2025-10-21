## Overview

This project implements a decision tree classifier from scratch using information gain for split selection. It includes tools for training, evaluation, cross-validation, and visualization of decision trees.

## Project Structure

```
├── DecisionTree.py          # Core decision tree implementation
├── CrossValidator.py        # K-fold cross-validation functionality
├── TreeEvaluator.py         # Model evaluation metrics and tools
├── TreeUtils.py             # Utility functions for tree operations
├── DrawTree.py              # Tree visualization utilities
├── main.py                  # Command-line interface
├── cw.ipynb                 # Jupyter notebook with analysis
├── requirements.txt         # Python package dependencies
└── wifi_db/                 # Dataset directory
    ├── clean_dataset.txt    # Clean WiFi signal data
    └── noisy_dataset.txt    # Noisy WiFi signal data
```

## Usage

### Command Line Interface

The project provides a simple CLI for training and evaluating decision trees:

#### Train a Decision Tree

Train a tree on training data and optionally evaluate on test data:

```bash
python main.py train --train-data wifi_db/clean_dataset.txt --test-data wifi_db/noisy_dataset.txt
```

You can also specify the maximum depth of the decision tree to be trained

```bash
python main.py train --train-data wifi_db/clean_dataset.txt --test-data wifi_db/noisy_dataset.txt --max-depth 100
```

#### K-Fold Cross-Validation

Perform k-fold cross-validation:

```bash
python main.py cross-validate --data wifi_db/clean_dataset.txt --k 10 --max-depth 1000
```

#### Cross-Validation with Test Evaluation

Perform cross-validation and evaluate the best tree on separate test data:

```bash
python main.py cross-validate --data wifi_db/clean_dataset.txt --test-data wifi_db/noisy_dataset.txt --k 10 --max-depth 1000
```

### Python API

You can also use the classes directly in your Python code:

```python
from DecisionTree import DecisionTree
from CrossValidator import KFoldCrossValidator
from TreeEvaluator import TreeEvaluator
import numpy as np

# Load data
data = np.loadtxt('wifi_db/clean_dataset.txt')

# Train a decision tree
tree = DecisionTree(data)
tree.train(max_depth=10)

# Evaluate the tree
test_data = np.loadtxt('wifi_db/noisy_dataset.txt')
accuracy = TreeEvaluator.evaluate(test_data, tree)
print(f"Accuracy: {accuracy}")

# Perform cross-validation
validator = KFoldCrossValidator(DecisionTree, k=10)
results = validator.k_fold_cross_validation(data)
print(f"Average CV Accuracy: {results['average_accuracy']}")
```

### Jupyter Notebook

The `cw.ipynb` notebook contains a complete analysis including:
- Data loading and preprocessing
- Model training and evaluation
- Cross-validation experiments
- Performance comparison between clean and noisy datasets
- Tree visualization and interpretation
- Feature importance analysis

## Key Classes

### DecisionTree
Core decision tree implementation with methods for:
- `train(max_depth=None)`: Train the tree with optional depth limit
- `predict(sample)`: Predict class for a single sample
- `batch_predict(samples)`: Predict classes for multiple samples
- `visualise(h_scaling, max_depth)`: Visualize the tree structure

### KFoldCrossValidator
Cross-validation implementation:
- `k_fold_cross_validation(data, tree_depth=None)`: Perform k-fold CV
- Returns best tree, average accuracy, and confusion matrix

### TreeEvaluator
Comprehensive evaluation metrics:
- `evaluate(test_data, tree)`: Calculate accuracy
- `confusion_matrix(test_data, tree)`: Generate confusion matrix
- `visualise_confusion_matrix(confusion_matrix, label_to_index, title)`: Returns a plot for the provided confusion matrix
- `get_precision_recall_per_class()`: Per-class precision/recall
- `get_f1_score()`: F1-score calculation

## Command Line Options

### Train Command
- `--train-data`: Path to training data file (required)
- `--test-data`: Path to test data file (optional)
- `--max-depth`: Maximum tree depth (optional, default: unlimited)

### Cross-Validate Command
- `--data`: Path to data file for cross-validation (required)
- `--test-data`: Path to test data for best tree evaluation (optional)
- `--k`: Number of folds (default: 10)
- `--max-depth`: Maximum tree depth (optional, default: unlimited)

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Jupyter (for notebook analysis)

See `requirements.txt` for specific versions.