import numpy as np
from DecisionTree import BaseDecisionTree

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

        accuracies = []
        folds = self.__k_fold_split(data)
        
        for i in range(self.k):
            # Use the i-th fold as the test set and the rest as the training set
            test_set = folds[i]
            train_set = np.vstack([folds[j] for j in range(self.k) if j != i])
            
            # creates a new decision tree instance based on the training set
            tree = self.tree_cls(train_set)
            tree.train(tree_depth)
            
            # Evaluate the decision tree on the test set
            accuracy = tree.evaluate(test_set)
            accuracies.append(accuracy)
            print(f"Fold {i + 1}: Accuracy = {accuracy:.4f}")
        
        average_accuracy = np.mean(accuracies)
        print(f"Average Accuracy over {self.k} folds: {average_accuracy:.4f}")
        return average_accuracy