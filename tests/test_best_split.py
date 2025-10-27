"""
Comprehensive tests for best split selection in decision tree.

These tests verify that:
1. The decision tree always selects the feature with maximum information gain
2. find_best_split correctly identifies the optimal feature and threshold
3. maximise_gain_for_feature finds the best threshold for each feature
4. The tree construction uses optimal splits at each node
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DecisionTree import DecisionTree
from TreeUtils import entropy, gain


class TestBestSplitSelection(unittest.TestCase):
    """Test that find_best_split selects the feature with maximum gain."""

    def test_find_best_split_returns_correct_structure(self):
        """Verify find_best_split returns (feature_index, threshold, gain)."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        
        feature_idx, threshold, info_gain = DecisionTree.find_best_split(data)
        
        self.assertIsNotNone(feature_idx, msg="Feature index should not be None")
        self.assertIsNotNone(threshold, msg="Threshold should not be None")
        self.assertIsInstance(feature_idx, (int, np.integer), msg="Feature index should be an integer")
        self.assertIsInstance(threshold, (float, np.floating), msg="Threshold should be a float")
        self.assertIsInstance(info_gain, (float, np.floating), msg="Info gain should be a float")

    def test_find_best_split_selects_maximum_gain(self):
        """Verify that find_best_split selects the feature with maximum gain."""
        # Create data where feature 0 perfectly separates, feature 1 doesn't
        data = np.array([
            [1.0, 10.0, 0],
            [1.1, 20.0, 0],
            [2.0, 10.0, 1],
            [2.1, 20.0, 1]
        ])
        
        # Calculate expected gains for each feature
        gain_f0, _ = DecisionTree.maximise_gain_for_feature(data, 0)
        gain_f1, _ = DecisionTree.maximise_gain_for_feature(data, 1)
        
        # Get the best split
        best_feature, _, best_gain = DecisionTree.find_best_split(data)
        
        # The best feature should give the maximum gain
        max_expected_gain = max(gain_f0, gain_f1)
        
        self.assertAlmostEqual(best_gain, max_expected_gain, places=10,
                              msg="Best split should select feature with maximum gain")
        
        # Verify it selected feature 0 (which has higher gain)
        if gain_f0 > gain_f1:
            self.assertEqual(best_feature, 0, msg="Should select feature 0 (higher gain)")
        else:
            self.assertEqual(best_feature, 1, msg="Should select feature 1 (higher gain)")

    def test_find_best_split_exhaustive_comparison(self):
        """Exhaustively verify that no other split has higher gain."""
        data = np.array([
            [1.0, 10.0, 0],
            [1.5, 15.0, 0],
            [2.0, 20.0, 1],
            [2.5, 25.0, 1],
            [3.0, 30.0, 2],
            [3.5, 35.0, 2]
        ])
        
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # Manually check all possible splits
        for feature_idx in range(data.shape[1] - 1):
            sorted_data = data[data[:, feature_idx].argsort()]
            feature_values = sorted_data[:, feature_idx]
            labels = sorted_data[:, -1]
            
            for i in range(1, len(feature_values)):
                if labels[i] != labels[i - 1]:
                    threshold = (feature_values[i] + feature_values[i - 1]) / 2
                    current_gain = gain(data, feature_idx, threshold)
                    
                    # No split should have gain greater than the best split
                    self.assertLessEqual(current_gain, best_gain + 1e-10,
                                       msg=f"Found split with higher gain: "
                                           f"feature {feature_idx}, threshold {threshold}, "
                                           f"gain {current_gain} > {best_gain}")

    def test_maximise_gain_for_feature_finds_best_threshold(self):
        """Verify maximise_gain_for_feature finds the best threshold for a feature."""
        data = np.array([
            [1.0, 0],
            [2.0, 0],
            [3.0, 1],
            [4.0, 1]
        ])
        
        feature_idx = 0
        best_gain, best_threshold = DecisionTree.maximise_gain_for_feature(data, feature_idx)
        
        # Test all possible thresholds manually
        sorted_data = data[data[:, feature_idx].argsort()]
        feature_values = sorted_data[:, feature_idx]
        labels = sorted_data[:, -1]
        
        for i in range(1, len(feature_values)):
            if labels[i] != labels[i - 1]:
                threshold = (feature_values[i] + feature_values[i - 1]) / 2
                current_gain = gain(data, feature_idx, threshold)
                
                # No threshold should give higher gain
                self.assertLessEqual(current_gain, best_gain + 1e-10,
                                   msg=f"Found threshold with higher gain: {threshold}, "
                                       f"gain {current_gain} > {best_gain}")

    def test_best_split_with_multiple_features_same_gain(self):
        """When multiple features have same gain, any valid choice is acceptable."""
        # Create data where both features separate classes equally
        data = np.array([
            [1.0, 1.0, 0],
            [2.0, 2.0, 1]
        ])
        
        gain_f0, _ = DecisionTree.maximise_gain_for_feature(data, 0)
        gain_f1, _ = DecisionTree.maximise_gain_for_feature(data, 1)
        
        best_feature, _, best_gain = DecisionTree.find_best_split(data)
        
        # Both features should have same gain
        self.assertAlmostEqual(gain_f0, gain_f1, places=10,
                              msg="Both features should have equal gain")
        
        # Best gain should match both
        self.assertAlmostEqual(best_gain, gain_f0, places=10,
                              msg="Best gain should match feature gains")


class TestDecisionTreeSplitOptimality(unittest.TestCase):
    """Test that DecisionTree actually uses optimal splits during training."""

    def test_tree_uses_best_split_at_root(self):
        """Verify the root node uses the optimal split."""
        # Create data where feature 0 is clearly better
        data = np.array([
            [1.0, 10.0, 0],
            [1.1, 20.0, 0],
            [2.0, 10.0, 1],
            [2.1, 20.0, 1]
        ])
        
        # Train tree
        tree = DecisionTree(data)
        tree.train()
        
        # Get the root split
        root_feature = tree.tree['feature_index']
        root_threshold = tree.tree['feature_threshold']
        
        # Calculate what the best split should be
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # Root should use the best split
        self.assertEqual(root_feature, best_feature,
                        msg="Root should use the feature with maximum gain")
        self.assertAlmostEqual(root_threshold, best_threshold, places=10,
                              msg="Root should use the optimal threshold")

    def test_tree_maximizes_gain_at_each_level(self):
        """Verify each internal node uses the optimal split for its subset."""
        data = np.array([
            [1.0, 10.0, 0],
            [1.2, 12.0, 0],
            [1.4, 14.0, 0],
            [2.0, 20.0, 1],
            [2.2, 22.0, 1],
            [2.4, 24.0, 1],
            [3.0, 30.0, 2],
            [3.2, 32.0, 2],
            [3.4, 34.0, 2]
        ])
        
        tree = DecisionTree(data)
        tree.train(max_depth=2)
        
        # Verify root split
        root_feature, root_threshold, root_gain = DecisionTree.find_best_split(data)
        self.assertEqual(tree.tree['feature_index'], root_feature,
                        msg="Root should use optimal feature")
        
        # Verify left child split (if it's a dict, not a leaf)
        left_data = data[data[:, root_feature] <= root_threshold]
        if isinstance(tree.tree['left_tree'], dict) and len(left_data) > 0:
            left_feature, left_threshold, left_gain = DecisionTree.find_best_split(left_data)
            self.assertEqual(tree.tree['left_tree']['feature_index'], left_feature,
                           msg="Left child should use optimal feature for its subset")
        
        # Verify right child split (if it's a dict, not a leaf)
        right_data = data[data[:, root_feature] > root_threshold]
        if isinstance(tree.tree['right_tree'], dict) and len(right_data) > 0:
            right_feature, right_threshold, right_gain = DecisionTree.find_best_split(right_data)
            self.assertEqual(tree.tree['right_tree']['feature_index'], right_feature,
                           msg="Right child should use optimal feature for its subset")

    def test_no_split_better_than_chosen_split(self):
        """For any node in the tree, verify no alternative split would be better."""
        np.random.seed(42)
        
        # Generate random data
        n_samples = 50
        data = np.random.randn(n_samples, 3)
        data[:, -1] = (data[:, 0] > 0).astype(int)
        
        tree = DecisionTree(data)
        tree.train(max_depth=3)
        
        # Helper function to verify a node uses optimal split
        def verify_node_optimality(node, node_data):
            if not isinstance(node, dict) or len(node_data) == 0:
                return True
            
            # Get the split used by this node
            node_feature = node['feature_index']
            node_threshold = node['feature_threshold']
            node_gain = gain(node_data, node_feature, node_threshold)
            
            # Find the best possible split for this data
            best_feature, best_threshold, best_gain = DecisionTree.find_best_split(node_data)
            
            # The node's gain should be very close to the best possible gain
            self.assertAlmostEqual(node_gain, best_gain, places=8,
                                  msg=f"Node uses suboptimal split: "
                                      f"gain={node_gain}, best_gain={best_gain}")
            
            # Recursively verify children
            left_data = node_data[node_data[:, node_feature] <= node_threshold]
            right_data = node_data[node_data[:, node_feature] > node_threshold]
            
            verify_node_optimality(node['left_tree'], left_data)
            verify_node_optimality(node['right_tree'], right_data)
        
        # Verify the entire tree
        verify_node_optimality(tree.tree, data)


class TestThresholdSelection(unittest.TestCase):
    """Test that thresholds are selected optimally."""

    def test_threshold_between_class_boundaries(self):
        """Thresholds should only be considered between samples of different classes."""
        data = np.array([
            [1.0, 0],
            [1.5, 0],
            [2.0, 0],
            [5.0, 1],
            [5.5, 1],
            [6.0, 1]
        ])
        
        _, threshold = DecisionTree.maximise_gain_for_feature(data, 0)
        
        # Threshold should be between 2.0 and 5.0 (boundary between classes)
        self.assertGreater(threshold, 2.0, msg="Threshold should be after last class 0 sample")
        self.assertLess(threshold, 5.0, msg="Threshold should be before first class 1 sample")
        
        # Specifically, it should be (2.0 + 5.0) / 2 = 3.5
        self.assertAlmostEqual(threshold, 3.5, places=10,
                              msg="Threshold should be midpoint between class boundaries")

    def test_optimal_threshold_maximizes_separation(self):
        """The selected threshold should maximize class separation."""
        data = np.array([
            [1.0, 0],
            [1.5, 0],
            [2.0, 0],
            [2.5, 1],
            [3.0, 0],
            [4.0, 1],
            [4.5, 1],
            [5.0, 1]
        ])
        
        best_gain, best_threshold = DecisionTree.maximise_gain_for_feature(data, 0)
        
        # Test that this threshold indeed gives the maximum gain
        sorted_data = data[data[:, 0].argsort()]
        feature_values = sorted_data[:, 0]
        labels = sorted_data[:, -1]
        
        for i in range(1, len(feature_values)):
            if labels[i] != labels[i - 1]:
                threshold = (feature_values[i] + feature_values[i - 1]) / 2
                current_gain = gain(data, 0, threshold)
                
                self.assertLessEqual(current_gain, best_gain + 1e-10,
                                   msg=f"Found better threshold: {threshold} with gain {current_gain}")


class TestEdgeCasesInSplitSelection(unittest.TestCase):
    """Test edge cases in split selection."""

    def test_pure_data_has_no_beneficial_split(self):
        """Pure data should have zero gain for any split."""
        data = np.array([
            [1.0, 0],
            [2.0, 0],
            [3.0, 0],
            [4.0, 0]
        ])
        
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # No split can improve a pure dataset
        # best_threshold might be None if no valid threshold found
        self.assertIsNone(best_threshold, 
                         msg="No valid threshold should exist for pure data")

    def test_binary_data_selects_separating_split(self):
        """With only two samples of different classes, should find separating split."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 1]
        ])
        
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # Should find a perfect split (gain = entropy of original data)
        original_entropy = entropy(data)
        self.assertAlmostEqual(best_gain, original_entropy, places=8,
                              msg="Perfect split should give gain equal to original entropy")

    def test_all_features_evaluated(self):
        """Verify that all features are considered during split selection."""
        # Create data where feature 2 is best
        data = np.array([
            [1.0, 1.0, 1.0, 0],  # feature 2 perfectly separates
            [1.5, 1.5, 1.5, 0],
            [2.0, 2.0, 5.0, 1],
            [2.5, 2.5, 5.5, 1]
        ])
        
        # Calculate gains for all features
        gains = []
        for feature_idx in range(3):
            g, _ = DecisionTree.maximise_gain_for_feature(data, feature_idx)
            gains.append(g)
        
        best_feature, _, best_gain = DecisionTree.find_best_split(data)
        
        # Best gain should be the maximum of all feature gains
        self.assertAlmostEqual(best_gain, max(gains), places=10,
                              msg="Best split should have maximum gain across all features")
        
        # Should select the feature with maximum gain
        max_gain_feature = np.argmax(gains)
        self.assertEqual(best_feature, max_gain_feature,
                        msg="Should select the feature with maximum gain")


class TestRealWorldScenarios(unittest.TestCase):
    """Test split selection with realistic data scenarios."""

    def test_wifi_data_scenario(self):
        """Test with WiFi signal strength data (realistic scenario)."""
        # WiFi signals: Strong signal in room 1, weak in room 2
        data = np.array([
            [-45, -50, -55, -52, 1],  # Room 1
            [-48, -52, -53, -50, 1],
            [-50, -49, -51, -48, 1],
            [-75, -80, -85, -78, 2],  # Room 2
            [-78, -82, -80, -76, 2],
            [-80, -79, -77, -81, 2]
        ])
        
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # Should find a high-gain split (close to perfect separation)
        original_entropy = entropy(data)
        self.assertGreater(best_gain, 0.9 * original_entropy,
                          msg="Should find high-quality split in WiFi data")
        
        # Verify no other split is better
        for feature_idx in range(4):
            gain_f, _ = DecisionTree.maximise_gain_for_feature(data, feature_idx)
            self.assertLessEqual(gain_f, best_gain + 1e-10,
                               msg=f"Feature {feature_idx} should not have higher gain than best split")

    def test_imbalanced_class_distribution(self):
        """Test split selection with imbalanced classes."""
        # 90% class 0, 10% class 1
        data = np.array([[i, 0] for i in range(90)] + [[i + 90, 1] for i in range(10)])
        
        best_feature, best_threshold, best_gain = DecisionTree.find_best_split(data)
        
        # Should still find the optimal split
        self.assertIsNotNone(best_feature, msg="Should find a split even with imbalanced data")
        self.assertIsNotNone(best_threshold, msg="Should find a threshold even with imbalanced data")
        self.assertGreater(best_gain, 0, msg="Should have positive gain with separable classes")


if __name__ == '__main__':
    unittest.main(verbosity=2)
