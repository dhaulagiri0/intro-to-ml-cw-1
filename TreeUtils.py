import numpy as np

# TODO test these utility functions
def entropy(data):
    """
    Calculate the entropy of the labels in the dataset.
    Returns a float.

    Parameters:
    - data: A 2D numpy array where the last column is the label.
    """

    labels, counts = np.unique(data[:, -1], return_counts=True)
    probabilities = counts / counts.sum()
    # Adding a small value to avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9)) 


def gain(data, feature_index, threshold): 
    """
    Calculate the information gain of a potential split on a given feature and threshold.
    Returns the information gain as a float.

    Parameters:
    - data: A 2D numpy array where the last column is the label.
    - feature_index: The index of the feature to split on.
    - threshold: The threshold value to split the feature.
    """

    # Split the data based on the threshold
    left_subset = data[data[:, feature_index] <= threshold]
    right_subset = data[data[:, feature_index] > threshold]
    
    # Calculate the entropy of the original dataset
    original_entropy = entropy(data)
    
    # Calculate the weighted entropy of the subsets
    # |left_subset| + |right_subset| = |data|
    left_entropy = entropy(left_subset)
    right_entropy = entropy(right_subset)
    weighted_entropy = (len(left_subset) / len(data)) * left_entropy + (len(right_subset) / len(data)) * right_entropy
    
    return original_entropy - weighted_entropy
