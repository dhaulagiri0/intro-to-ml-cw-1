#!/usr/bin/env python3
"""
Command-line interface for Decision Tree classification.

This script provides functionality to:
1. Train a decision tree on a dataset
2. Evaluate a trained tree on test data
3. Perform k-fold cross-validation
4. Evaluate the best tree from cross-validation on test data

Usage examples:
    # Train a tree and evaluate on test data
    python main.py train --train-data wifi_db/clean_dataset.txt --test-data wifi_db/noisy_dataset.txt --max-depth 10

    # Perform 10-fold cross-validation
    python main.py cross-validate --data wifi_db/clean_dataset.txt --k 10 --max-depth 1000

    # Perform cross-validation and evaluate best tree on test data
    python main.py cross-validate --data wifi_db/clean_dataset.txt --test-data wifi_db/noisy_dataset.txt --k 10 --max-depth 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from CrossValidator import KFoldCrossValidator
from DecisionTree import DecisionTree
from TreeEvaluator import TreeEvaluator


def load_data(file_path):
    """
    Load data from a text file and ensure labels are integers.
    Returns a 2D numpy array where the last column contains integer labels

    Parameters:
    - file_path: Path to the data file
    """
    try:
        data = np.loadtxt(file_path)
        # Convert labels to integers
        labels = data[:, -1].astype(int)
        return np.column_stack((data[:, :-1], labels))
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        sys.exit(1)


def evaluate_tree_on_test_data(tree, test_data, description="Test"):
    """
    Evaluate a trained decision tree on test data and display detailed metrics.

    Parameters:
    - tree: Trained DecisionTree instance
    - test_data: 2D numpy array with features and labels
    - description: String description for the evaluation (e.g., "Test", "Validation")
    """
    print(f"\n{description} data shape: {test_data.shape}")

    # Calculate accuracy
    accuracy = TreeEvaluator.evaluate(test_data, tree)
    print(f"{description} accuracy: {accuracy:.4f}")

    # Show confusion matrix
    confusion_matrix = TreeEvaluator.confusion_matrix(test_data, tree)
    print(f"\n{description} Confusion Matrix:")
    print(confusion_matrix)

    # Calculate precision, recall, F1 per class
    precision_dict, recall_dict = TreeEvaluator.get_precision_recall_per_class(
        confusion_matrix, tree.label_to_index
    )

    print(f"\n{description} per-class metrics:")
    for label in sorted(tree.label_to_index.keys()):
        precision = precision_dict[label]
        recall = recall_dict[label]
        f1 = TreeEvaluator.get_f1_score(precision, recall)
        print(
            f"Class {int(label)}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )


def train_and_evaluate(args):
    """Train a decision tree and evaluate it on test data."""
    print("Loading training data...")
    train_data = load_data(args.train_data)
    print(f"Training data shape: {train_data.shape}")

    print("Training decision tree...")
    tree = DecisionTree(train_data)
    tree.train(max_depth=args.max_depth)
    print(f"Tree trained with depth: {tree.depth}")

    # Evaluate on training data
    train_accuracy = TreeEvaluator.evaluate(train_data, tree)
    print(f"Training accuracy: {train_accuracy:.4f}")

    if args.test_data:
        print("Loading test data...")
        test_data = load_data(args.test_data)
        evaluate_tree_on_test_data(tree, test_data, description="Test")


def cross_validate(args):
    """Perform k-fold cross-validation."""
    print(f"Loading data for {args.k}-fold cross-validation...")
    data = load_data(args.data)
    print(f"Data shape: {data.shape}")

    print(f"Performing {args.k}-fold cross-validation...")
    validator = KFoldCrossValidator(DecisionTree, args.k)

    # Perform cross-validation
    results = validator.k_fold_cross_validation(data, tree_depth=args.max_depth)

    # Display results
    print(f"\nCross-validation Results:")
    print(f"Average accuracy: {results['average_accuracy']:.4f}")

    # Calculate per-class metrics from average confusion matrix
    confusion_matrix = results["avg_confusion_matrix"]
    label_to_index = results["label_to_index"]
    precision_dict, recall_dict = TreeEvaluator.get_precision_recall_per_class(
        confusion_matrix, label_to_index
    )

    print("\nAverage Confusion Matrix:")
    print(confusion_matrix)

    print("\nAverage per-class metrics:")
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    num_classes = len(label_to_index)

    for label in sorted(label_to_index.keys()):
        precision = precision_dict[label]
        recall = recall_dict[label]
        f1 = TreeEvaluator.get_f1_score(precision, recall)
        print(
            f"Class {int(label)}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

    print(f"\nOverall averages:")
    print(f"Average Precision: {avg_precision/num_classes:.4f}")
    print(f"Average Recall: {avg_recall/num_classes:.4f}")
    print(f"Average F1: {avg_f1/num_classes:.4f}")

    # If test data is provided, evaluate the best tree on it
    if args.test_data:
        print(f"\nEvaluating best tree on test data...")
        test_data = load_data(args.test_data)

        best_tree = results["best_tree"]
        print(f"Best tree depth: {best_tree.depth}")

        evaluate_tree_on_test_data(best_tree, test_data, "Test (Best Tree)")


def main():
    parser = argparse.ArgumentParser(
        description="Decision Tree Classification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train", help="Train and evaluate a decision tree"
    )
    train_parser.add_argument(
        "--train-data", required=True, type=str, help="Path to training data file"
    )
    train_parser.add_argument(
        "--test-data", type=str, help="Path to test data file (optional)"
    )
    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of the tree (default: no limit)",
    )

    # Cross-validation command
    cv_parser = subparsers.add_parser(
        "cross-validate", help="Perform k-fold cross-validation"
    )
    cv_parser.add_argument(
        "--data", required=True, type=str, help="Path to data file for cross-validation"
    )
    cv_parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data file to evaluate best tree (optional)",
    )
    cv_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of folds for cross-validation (default: 10)",
    )
    cv_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of the trees (default: no limit)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate file paths
    if hasattr(args, "train_data") and args.train_data:
        if not Path(args.train_data).exists():
            print(f"Error: Training data file '{args.train_data}' not found.")
            sys.exit(1)

    if hasattr(args, "data") and args.data:
        if not Path(args.data).exists():
            print(f"Error: Data file '{args.data}' not found.")
            sys.exit(1)

    if hasattr(args, "test_data") and args.test_data:
        if not Path(args.test_data).exists():
            print(f"Error: Test data file '{args.test_data}' not found.")
            sys.exit(1)

    # Execute the appropriate command
    try:
        if args.command == "train":
            train_and_evaluate(args)
        elif args.command == "cross-validate":
            cross_validate(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
