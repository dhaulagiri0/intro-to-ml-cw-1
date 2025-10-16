from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from DrawTree import Tree, buchheim
from TreeUtils import gain


class BaseDecisionTree(ABC):
    """
    An abstract base class for decision tree implementations.
    """

    ## Constants for dictionary keys
    FEATURE_INDEX: str = "feature_index"
    FEATURE_THRESHOLD: str = "feature_threshold"
    LEFT_TREE: str = "left_tree"
    RIGHT_TREE: str = "right_tree"

    @abstractmethod
    def __init__(self, data, depth=None):
        pass

    @abstractmethod
    def train(self, max_depth=None):
        """
        Train the decision tree on the provided dataset.
        We expect each instance of the decision tree to be initialised with the training data, 
        which cannot be modified later.
        This makes sense since the tree trained is tightly coupled with the data it is trained on.

        Parameters:
        - max_depth: The maximum depth of the tree. If None, the tree will grow until all leaves are pure.
        """
        pass

    @abstractmethod
    def evaluate(self, test_data):
        """
        Evaluate the decision tree on the provided test dataset.
        Returns the accuracy as a float.

        Parameters:
        - test_data: A 2D numpy array where the last column is the label.
        """
        pass

    @abstractmethod
    def predict(self, sample):
        """
        Predict the label for a given sample (without label).
        Returns the predicted label.

        Parameters:
        - sample: A 1D numpy array representing the features of the sample.
        """
        pass

    @abstractmethod
    def visualise(self, x: int, max_depth=5) -> plt:
        """
        Visualise the decision tree structure, with pure matplotlib.
        Returns the matplotlib.pyplot object for further manipultion or display.

        Parameters:
        - h_scaling: The horizontal scaling factor for the visualization, larger means wider plot.
        - max_depth: The maximum depth to visualize.
        """
        pass


class DecisionTree(BaseDecisionTree):
    """
    A decision tree implementation that tries to maximise information gain for each decision.
    """

    depth = 0  # Track the depth of the tree

    def __init__(self, data):
        self.data = data

    def train(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = self.__create_decision_tree(self.data)


    ## Helper functions for decision tree creation
    def __create_decision_tree(self, current_data, current_depth=0):
        """
        Recursively create the decision tree, up to a maximum depth if specified.
        Returns the tree as a nested dictionary.

        Parameters:
        - current_data: A 2D numpy array where the last column is the label.
        - current_depth: The current depth of the tree.
        """

        # Base case: if depth is 0 or data is empty, return None
        if len(current_data) == 0 or (self.max_depth is not None and current_depth == self.max_depth):
            return None

        # If all labels are the same, return that label
        if np.all(current_data[:, -1] == current_data[0, -1]):
            return current_data[0, -1]

        feature_index, feature_threshold, info_gain = DecisionTree.find_best_split(current_data)

        # Split the data based on the selected feature
        left_subset = current_data[current_data[:, feature_index] <= feature_threshold]
        right_subset = current_data[current_data[:, feature_index] > feature_threshold]
        left_tree = self.__create_decision_tree(left_subset, current_depth + 1)
        right_tree = self.__create_decision_tree(right_subset, current_depth + 1)

        self.depth = max(current_depth + 1, self.depth)
        return {
            BaseDecisionTree.FEATURE_INDEX: feature_index,
            BaseDecisionTree.FEATURE_THRESHOLD: feature_threshold,
            BaseDecisionTree.LEFT_TREE: left_tree,
            BaseDecisionTree.RIGHT_TREE: right_tree
        }


    @staticmethod
    def maximise_gain_for_feature(data, feature_index):
        """
        For a given feature, find the threshold that maximises information gain.
        Returns the best gain and the corresponding threshold.
        Parameters:
        - data: A 2D numpy array where the last column is the label.
        - feature_index: The index of the feature to evaluate.
        """

        best_gain = -1
        best_threshold = None

        # sort the data by the feature to consider values between entries of different classes
        sorted_data = data[data[:, feature_index].argsort()]
        feature_values = sorted_data[:, feature_index]
        labels = sorted_data[:, -1]
        # find midpoints between different class labels
        for i in range(1, len(feature_values)):
            if labels[i] != labels[i - 1]:  # Only consider thresholds between different classes
                threshold = (feature_values[i] + feature_values[i - 1]) / 2
                current_gain = gain(data, feature_index, threshold)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_threshold = threshold

        return best_gain, best_threshold


    @staticmethod
    def find_best_split(data):
        """
        Find the best feature and feature value to split on to maximise information gain.
        Returns the best feature index, best threshold, and the corresponding information gain.

        Parameters:
        - data: A 2D numpy array where the last column is the label.
        """

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature_index in range(data.shape[1] - 1):  # Exclude the label column
            current_gain, current_threshold = DecisionTree.maximise_gain_for_feature(
                data, feature_index)
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feature_index
                best_threshold = current_threshold

        return best_feature, best_threshold, best_gain


    def evaluate(self, test_data):
        """
        Evaluate the decision tree on the provided test dataset.
        Returns the accuracy as a float.

        Parameters:
        - test_data: A 2D numpy array where the last column is the label.
        """
        X = test_data[:, :-1]
        y_true = test_data[:, -1]
        y_pred = np.array([self.predict(x) for x in X])
        return np.mean(y_pred == y_true)


    def predict(self, sample):
        if self.tree is None:
            raise ValueError("The decision tree has not been trained yet. " \
            "Please call the 'train' method before prediction.")
        return self.__predict(self.tree, sample)


    def __predict(self, current_tree, sample):
        """
        Recursively traverse the decision tree to predict the label for a given sample.
        Returns the predicted label.

        Parameters:
        - current_tree: The current node of the decision tree (can be a dict or a
          label if it's a leaf).
        """

        # Check that the sample has the correct number of features
        if len(sample) != self.data.shape[1] - 1:
            raise ValueError("Sample has incorrect number of features")

        # If the tree is a leaf node, return the label
        if not isinstance(current_tree, dict):
            return current_tree

        feature_index = current_tree[BaseDecisionTree.FEATURE_INDEX]
        feature_threshold = current_tree[BaseDecisionTree.FEATURE_THRESHOLD]

        if sample[feature_index] <= feature_threshold:
            return self.__predict(current_tree[BaseDecisionTree.LEFT_TREE], sample)
        else:
            return self.__predict(current_tree[BaseDecisionTree.RIGHT_TREE], sample)


    def visualise(self, h_scaling: int, max_depth=5) -> plt:
        if self.tree is None:
            raise ValueError("The decision tree has not been trained yet. "
            "Please call the 'train' method before visualization.")

        _, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        tree = self.__parse_draw_tree(self.tree, 0)
        draw_tree = buchheim(tree)

        def draw_node(node, depth):
            if depth >= max_depth: return
            if node is not None:
                ax.text(node.x * h_scaling, -node.y * h_scaling, node.label, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1),
                        fontsize=5)
                if node.parent is not None:
                    ax.plot(
                        [node.x * h_scaling, node.parent.x * h_scaling], 
                        [-node.y * h_scaling, -node.parent.y * h_scaling], 
                        'k-')
                for child in node.children:
                    draw_node(child, depth + 1)

        draw_node(draw_tree, 0)
        return plt


    def __parse_draw_tree(self, current_tree, node_id):
        """
        Recursively parse the decision tree into a format suitable for visualization.
        Returns a Tree object representing the structure of the decision tree.

        Parameters:
        - current_tree: The current node of the decision tree (can be a dict or a
          label if it's a leaf).
        - node_id: An integer representing the unique ID of the current node.
        """

        if not isinstance(current_tree, dict):
            return Tree(str(current_tree), node_id)
        left_tree = self.__parse_draw_tree(current_tree[BaseDecisionTree.LEFT_TREE], 0)
        right_tree = self.__parse_draw_tree(current_tree[BaseDecisionTree.RIGHT_TREE], 1)
        return Tree(
            f"[X{current_tree[BaseDecisionTree.FEATURE_INDEX]} < "
            f"{current_tree[BaseDecisionTree.FEATURE_THRESHOLD]:.2f}]", 
            node_id, left_tree, right_tree)
