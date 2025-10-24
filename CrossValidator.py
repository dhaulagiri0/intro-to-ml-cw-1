import numpy as np

from DecisionTree import BaseDecisionTree
from TreeEvaluator import TreeEvaluator


class KFoldCrossValidator:
    """
    A class to perform k-fold cross-validation on a given decision tree implementation.

    Constructor Parameters:
    - tree_cls: A class that implements the BaseDecisionTree ABC.
    - k: Number of folds for cross-validation.
    """

    def __init__(self, tree_cls: BaseDecisionTree, k: int):
        self.k = k
        self.tree_cls = tree_cls


    def __k_fold_split(self, data):
        """
        Splits the data into k folds for cross-validation.
        The output of this function is non-deterministic due to random shuffling of the
        provided dataset.

        Parameters:
        - data: A 2D numpy array where the last column is the label.
        """

        np.random.shuffle(data)
        ## We might not be able to split evenly, so we will round off here and add any leftover
        # samples to the last fold
        fold_size = len(data) // self.k
        folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(self.k)]

        # Handle any leftover samples
        if len(data) % self.k != 0:
            folds[-1] = np.vstack((folds[-1], data[self.k * fold_size:]))

        return folds


    def k_fold_cross_validation(self, data, tree_depth=None):
        """
        Perform k-fold cross-validation on the provided dataset.
        For each fold, train a new decision tree on the training set and evaluate it on the
        test set. Returns the average accuracy across all folds.

        Parameters:
        - data: A 2D numpy array where the last column is the label.
        - tree_depth: Optional maximum depth for the decision tree.
        """

        folds = self.__k_fold_split(data)
        max_accuracy = 0
        best_tree = None
        cumulative_accuracy = 0
        cumulative_confusion_matrix = None
        for i in range(self.k):
            # Use the i-th fold as the test set and the rest as the training set
            test_set = folds[i]
            train_set = np.vstack([folds[j] for j in range(self.k) if j != i])

            # creates a new decision tree instance based on the training set
            tree = self.tree_cls(train_set)
            tree.train(tree_depth)

            # Evaluate the decision tree on the test set
            accuracy = TreeEvaluator.evaluate(test_set, tree)
            confusion_matrix = TreeEvaluator.confusion_matrix(
                test_set,
                tree=tree)
            cumulative_confusion_matrix = (
                confusion_matrix if cumulative_confusion_matrix is None
                else cumulative_confusion_matrix + confusion_matrix
            )
            print(f"Fold {i + 1}: Accuracy = {accuracy:.4f}")

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_tree = tree
            cumulative_accuracy += accuracy

        return {
            "average_accuracy": cumulative_accuracy / self.k,
            "best_tree": best_tree,
            "avg_confusion_matrix": cumulative_confusion_matrix / self.k,
            "label_to_index": tree.label_to_index,
        }
