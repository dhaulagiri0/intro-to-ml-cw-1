import numpy as np
import matplotlib.pyplot as plt

from DecisionTree import DecisionTree


class TreeEvaluator:

    @staticmethod
    def evaluate(test_db: np.array, tree: DecisionTree) -> float:
        """
        Evaluate the provided decision tree on the test dataset.
        Returns the accuracy as a float.

        Parameters:
        - test_db: A 2D numpy array where the last column is the label.
        - tree: An instance of DecisionTree to evaluate.
        """

        ## Ensure that the tree has been trained
        if tree.tree is None:
            raise ValueError("The decision tree must be trained before evaluation.")

        y_true = test_db[:, -1]
        y_pred = tree.batch_predict(test_db[:, :-1])
        return np.mean(y_pred == y_true)

    @staticmethod
    def confusion_matrix(test_db: np.array, tree: DecisionTree) -> np.array:
        """
        Calculate the confusion matrix for the provided decision tree on the test dataset.

        Parameters:
        - test_db: A 2D numpy array where the last column is the label.
        - tree: An instance of DecisionTree to evaluate.

        Returns:
        - A 2D numpy array representing the confusion matrix.
        """

        ## Ensure that the tree has been trained
        if tree.tree is None:
            raise ValueError("The decision tree must be trained before evaluation.")

        num_classes = len(tree.label_to_index)
        y_true = test_db[:, -1]
        y_pred = tree.batch_predict(test_db[:, :-1])

        true_labels = y_true
        predicted_labels = y_pred

        # If predicted_labels contains None values, this will raise an error
        assert None not in predicted_labels, "Predicted labels contain None values."

        # Ensure all classes in true_labels are in predicted_labels
        for label in np.unique(true_labels):
            if label not in np.unique(predicted_labels):
                raise ValueError(
                    f"Predicted labels must contain all classes present in true labels. Missing class: {label}"
                )

        # Calculate confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(true_labels, predicted_labels):
            confusion_matrix[tree.label_to_index[t], tree.label_to_index[p]] += 1
        return confusion_matrix

    @staticmethod
    def average_confusion_per_class(confusion_matrix: np.array) -> float:
        """
        Calculate the average confusion per class from the confusion matrix.

        Parameters:
        - confusion_matrix: A 2D numpy array representing the confusion matrix.

        Returns:
        - A float representing the average confusion per class.
        """

        num_classes = confusion_matrix.shape[0]
        rate_of_confusion_dict = {}
        total_confusion = 0

        for i in range(num_classes):
            true_positives = confusion_matrix[i, i]
            total_instances = np.sum(confusion_matrix[i, :])
            false_negatives = total_instances - true_positives
            total_confusion += false_negatives

        return total_confusion / num_classes if num_classes > 0 else 0.0

    @staticmethod
    def get_accuracy(confusion_matrix: np.array) -> float:
        """
        Calculate accuracy from the confusion matrix.

        Parameters:
        - confusion_matrix: A 2D numpy array representing the confusion matrix.

        Returns:
        - A float representing the accuracy.
        """

        correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    @staticmethod
    def get_precision_recall_per_class(
        confusion_matrix: np.array, label_to_index: dict
    ) -> tuple:
        """
        Calculate precision and recall for each class from the confusion matrix.

        Parameters:
        - confusion_matrix: A 2D numpy array representing the confusion matrix.
        - label_to_index: A dictionary mapping labels to their corresponding indices in the confusion matrix.

        Returns:
        - A tuple of two dictionaries: (precision_dict, recall_dict)
        """

        # Invert the label_to_index mapping
        index_to_label = {index: label for label, index in label_to_index.items()}

        num_classes = confusion_matrix.shape[0]
        precision_dict = {}
        recall_dict = {}

        for i in range(num_classes):
            true_positive = confusion_matrix[i, i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive

            # We have to be careful about division by zero
            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0.0
            )

            precision_dict[index_to_label[i]] = precision
            recall_dict[index_to_label[i]] = recall

        # avg precision and recall
        precision_dict["avg"] = np.mean(list(precision_dict.values()))
        recall_dict["avg"] = np.mean(list(recall_dict.values()))
        return (precision_dict, recall_dict)

    @staticmethod
    def get_f1_score(precision: float, recall: float) -> float:
        """
        Calculate the F1 score given precision and recall.

        Parameters:
        - precision: A float representing precision.
        - recall: A float representing recall.

        Returns:
        - A float representing the F1 score.
        """

        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def visualise_confusion_matrix(
        confusion_matrix: np.array, label_to_index: dict, title: str
    ) -> plt:
        labels = list(label_to_index.keys())

        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)

        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    confusion_matrix[i, j],
                    ha="center",
                    va="center",
                    color="black",
                )

        return plt
