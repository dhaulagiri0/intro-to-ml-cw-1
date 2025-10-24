from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from DrawTree import buchheim
from TreeUtils import entropy, gain


class BaseDecisionTree(ABC):
    """
    An abstract base class for decision tree implementations.
    """

    ## Constants for dictionary keys
    FEATURE_INDEX: str = "feature_index"
    FEATURE_THRESHOLD: str = "feature_threshold"
    LEFT_TREE: str = "left_tree"
    RIGHT_TREE: str = "right_tree"
    tree = None
    label_to_index = None
    feature_importance: dict = None

    @abstractmethod
    def __init__(self, data, depth=None, label_to_index=None):
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
    def predict(self, sample) -> "BaseDecisionTree":
        """
        Predict the label for a given sample (without label).
        Returns the predicted label.

        Parameters:
        - sample: A 1D numpy array representing the features of the sample.
        """
        pass

    @abstractmethod
    def batch_predict(self, samples) -> np.ndarray:
        """
        Predict the labels for a batch of samples.
        Returns a 1D numpy array of predicted labels.

        Parameters:
        - samples: A 2D numpy array where each row represents the features of a sample.
        """
        pass

    @abstractmethod
    def visualise(self, h_scaling: int, max_depth=5):
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

    def __init__(self, data, label_to_index=None):
        self.data = data
        if label_to_index is None:
            self.label_to_index = {
                label: idx for idx, label in enumerate(np.unique(data[:, -1]))
            }
        else:
            self.label_to_index = label_to_index
        self.feature_importance = {}

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

        # If data is empty, return None
        if len(current_data) == 0:
            return None

        if self.max_depth is not None and current_depth >= self.max_depth:
            # Return the majority label in the current data
            labels, counts = np.unique(current_data[:, -1], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            return majority_label

        # If all labels are the same, return that label
        if np.all(current_data[:, -1] == current_data[0, -1]):
            return current_data[0, -1]

        feature_index, feature_threshold, info_gain = DecisionTree.find_best_split(
            current_data
        )
        # Store feature importance
        # The weight is the proportion of samples at this node relative to the total samples
        self.__update_feature_importance(
            feature_index, info_gain, len(current_data) / len(self.data)
        )

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
            BaseDecisionTree.RIGHT_TREE: right_tree,
        }

    def __update_feature_importance(
        self, feature_index, info_gain: float, weight: float
    ):
        """
        Update the feature importance dictionary with the given feature index,
        entropy, and weight. Using information gain as a measure of importance.

        Parameters:
        - feature_index: The index of the feature.
        - entropy: The entropy value associated with the feature split.
        - weight: A weight factor (e.g., proportion of samples at this node)."""
        if feature_index not in self.feature_importance:
            self.feature_importance[feature_index] = info_gain * weight
        else:
            self.feature_importance[feature_index] += info_gain * weight

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
            if (
                labels[i] != labels[i - 1]
            ):  # Only consider thresholds between different classes
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
                data, feature_index
            )
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feature_index
                best_threshold = current_threshold

        return best_feature, best_threshold, best_gain

    def predict(self, sample):
        if self.tree is None:
            raise ValueError(
                "The decision tree has not been trained yet. "
                "Please call the 'train' method before prediction."
            )
        return self.__predict(self.tree, sample)


    def batch_predict(self, samples) -> np.ndarray:
        if self.tree is None:
            raise ValueError(
                "The decision tree has not been trained yet. "
                "Please call the 'train' method before prediction."
            )
        return np.array([self.__predict(self.tree, sample) for sample in samples])

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
            raise ValueError(
                "The decision tree has not been trained yet. "
                "Please call the 'train' method before visualization."
            )

        draw_tree = buchheim(self.tree)

        _, ax = plt.subplots(figsize=(12 * h_scaling, 8))
        ax.axis("off")

        def draw_node(node, depth):
            if depth >= max_depth:
                return
            if node is not None:
                ax.text(
                    node.x * h_scaling,
                    -node.y * h_scaling,
                    node.label,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
                    fontsize=5,
                )
                if node.parent is not None:
                    ax.plot(
                        [node.x * h_scaling, node.parent.x * h_scaling],
                        [-node.y * h_scaling, -node.parent.y * h_scaling],
                        "k-",
                    )
                for child in node.children:
                    draw_node(child, depth + 1)

        draw_node(draw_tree, 0)
        return plt

    def get_top_k_features(self, k: int) -> list[tuple[int, float]]:
        """
        Get the top k important features based on feature importance scores.
        Returns a list of (feature indices, importance scores).

        Parameters:
        - k: The number of top features to return.
        """
        # Sort features by importance score in descending order
        sorted_features = sorted(
            self.feature_importance.items(), key=lambda item: item[1], reverse=True
        )

        # Get the top k feature indices and their importance scores
        top_k_features = [(feature[0], feature[1]) for feature in sorted_features[:k]]

        return top_k_features
