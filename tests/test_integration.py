"""
Integration tests for entropy and gain functions working together.

Tests the relationship between entropy and gain calculations,
and validates that they work correctly in realistic decision tree scenarios.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TreeUtils import entropy, gain


class TestEntropyGainIntegration(unittest.TestCase):
    """Integration tests for entropy and gain."""

    def test_gain_decomposition(self):
        """Verify that gain = H(parent) - weighted_H(children)."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        
        threshold = 2.5
        feature_idx = 0
        
        # Calculate gain using the function
        calculated_gain = gain(data, feature_idx, threshold)
        
        # Calculate manually
        parent_entropy = entropy(data)
        left_data = data[data[:, feature_idx] <= threshold]
        right_data = data[data[:, feature_idx] > threshold]
        
        left_entropy = entropy(left_data)
        right_entropy = entropy(right_data)
        
        weighted_child_entropy = (len(left_data) / len(data)) * left_entropy + \
                                 (len(right_data) / len(data)) * right_entropy
        
        manual_gain = parent_entropy - weighted_child_entropy
        
        self.assertAlmostEqual(calculated_gain, manual_gain, places=10,
                              msg="Gain should equal parent entropy minus weighted child entropy")

    def test_perfect_split_gives_maximum_gain(self):
        """A perfect split should give gain equal to parent entropy."""
        # Create perfectly separable data
        data = np.array([
            [1.0, 0.0, 0],
            [1.1, 0.1, 0],
            [1.2, 0.2, 0],
            [2.0, 1.0, 1],
            [2.1, 1.1, 1],
            [2.2, 1.2, 1]
        ])
        
        parent_entropy = entropy(data)
        calculated_gain = gain(data, feature_index=0, threshold=1.5)
        
        # Perfect split should give gain equal to parent entropy
        self.assertAlmostEqual(calculated_gain, parent_entropy, places=7,
                              msg="Perfect split should give gain equal to parent entropy")

    def test_no_split_improvement(self):
        """A split that doesn't improve purity should give near-zero gain."""
        # Create data where split doesn't help
        data = np.array([
            [1.0, 0.0, 0],
            [1.1, 0.1, 1],
            [2.0, 1.0, 0],
            [2.1, 1.1, 1]
        ])
        
        # Split that maintains same distribution on both sides
        calculated_gain = gain(data, feature_index=0, threshold=1.5)
        
        self.assertAlmostEqual(calculated_gain, 0.0, places=5,
                              msg="Split with no improvement should give near-zero gain")

    def test_entropy_gain_relationship_with_wifi_data(self):
        """Test with realistic WiFi signal strength scenario."""
        # Simulate WiFi signal data: signal strength vs room
        # Strong signal (> -60 dBm) in room 1, weak signal (<= -60 dBm) in room 2
        data = np.array([
            [-45, -50, -55, -52, 1],  # Room 1 - strong signals
            [-48, -52, -53, -50, 1],
            [-50, -49, -51, -48, 1],
            [-75, -80, -85, -78, 2],  # Room 2 - weak signals
            [-78, -82, -80, -76, 2],
            [-80, -79, -77, -81, 2]
        ])
        
        parent_entropy = entropy(data)
        
        # Test split on first WiFi signal
        threshold = -60
        calculated_gain = gain(data, feature_index=0, threshold=threshold)
        
        # This should be a perfect split
        self.assertAlmostEqual(calculated_gain, parent_entropy, places=7,
                              msg="WiFi signal split should give high gain")

    def test_multi_feature_gain_comparison(self):
        """Compare gains across multiple features to find best split."""
        # Create data where feature 1 is better than feature 0
        data = np.array([
            [1.0, 10.0, 0],
            [2.0, 10.5, 0],
            [1.5, 20.0, 1],
            [2.5, 20.5, 1]
        ])
        
        gain_feature_0 = gain(data, feature_index=0, threshold=1.75)
        gain_feature_1 = gain(data, feature_index=1, threshold=15.0)
        
        # Feature 1 should give better gain
        self.assertGreater(gain_feature_1, gain_feature_0,
                          msg="Feature 1 should provide better split than feature 0")

    def test_iterative_splitting_reduces_entropy(self):
        """Simulate decision tree building: entropy should decrease with each split."""
        # Start with mixed data
        data = np.array([
            [1.0, 10.0, 0],
            [1.5, 10.5, 0],
            [2.0, 20.0, 1],
            [2.5, 20.5, 1],
            [3.0, 30.0, 2],
            [3.5, 30.5, 2]
        ])
        
        initial_entropy = entropy(data)
        
        # First split on feature 0
        threshold_1 = 1.75
        gain_1 = gain(data, feature_index=0, threshold=threshold_1)
        
        # Calculate entropy after split
        left_1 = data[data[:, 0] <= threshold_1]
        right_1 = data[data[:, 0] > threshold_1]
        
        weighted_entropy_1 = (len(left_1) / len(data)) * entropy(left_1) + \
                            (len(right_1) / len(data)) * entropy(right_1)
        
        # Weighted entropy should be less than initial
        self.assertLess(weighted_entropy_1, initial_entropy,
                       msg="Split should reduce weighted entropy")
        
        # Gain should equal the reduction
        self.assertAlmostEqual(gain_1, initial_entropy - weighted_entropy_1, places=10)

    def test_weighted_entropy_calculation(self):
        """Verify weighted entropy calculation in gain."""
        data = np.array([
            [1.0, 0.0, 0],
            [1.0, 0.0, 0],
            [1.0, 0.0, 0],
            [2.0, 0.0, 1]
        ])
        
        threshold = 1.5
        feature_idx = 0
        
        calculated_gain = gain(data, feature_idx, threshold)
        
        # Manual calculation
        left_data = data[data[:, feature_idx] <= threshold]
        right_data = data[data[:, feature_idx] > threshold]
        
        # Left is pure (3 samples, all class 0) -> entropy = 0
        # Right is pure (1 sample, class 1) -> entropy = 0
        # Weighted entropy should be 0
        
        parent_entropy = entropy(data)  # Should be H(0.75, 0.25)
        
        self.assertAlmostEqual(calculated_gain, parent_entropy, places=7,
                              msg="Perfect split should give gain equal to parent entropy")

    def test_entropy_conservation(self):
        """Total information is conserved: H(parent) = gain + weighted_H(children)."""
        np.random.seed(42)
        
        # Generate random data
        n_samples = 50
        data = np.random.randn(n_samples, 3)
        data[:, -1] = (data[:, 0] > 0).astype(int)
        
        threshold = 0.0
        feature_idx = 0
        
        parent_entropy = entropy(data)
        calculated_gain = gain(data, feature_idx, threshold)
        
        left_data = data[data[:, feature_idx] <= threshold]
        right_data = data[data[:, feature_idx] > threshold]
        
        weighted_child_entropy = (len(left_data) / len(data)) * entropy(left_data) + \
                                 (len(right_data) / len(data)) * entropy(right_data)
        
        # Parent entropy = gain + weighted child entropy
        self.assertAlmostEqual(parent_entropy, calculated_gain + weighted_child_entropy,
                              places=10,
                              msg="Entropy should be conserved")


class TestRealisticScenarios(unittest.TestCase):
    """Test with realistic decision tree scenarios."""

    def test_iris_like_scenario(self):
        """Test with Iris-like data (3 classes, multiple features)."""
        # Simplified Iris-like scenario
        data = np.array([
            [5.1, 3.5, 0],  # Setosa
            [4.9, 3.0, 0],
            [5.0, 3.6, 0],
            [6.5, 2.8, 1],  # Versicolor
            [6.7, 3.0, 1],
            [6.3, 2.5, 1],
            [7.2, 3.6, 2],  # Virginica
            [7.4, 2.8, 2],
            [7.9, 3.8, 2]
        ])
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(2):
            for threshold in np.linspace(data[:, feature_idx].min(), 
                                       data[:, feature_idx].max(), 10):
                g = gain(data, feature_idx, threshold)
                if g > best_gain:
                    best_gain = g
                    best_feature = feature_idx
                    best_threshold = threshold
        
        self.assertIsNotNone(best_feature, msg="Should find a best feature")
        self.assertGreater(best_gain, 0, msg="Best split should have positive gain")

    def test_xor_problem(self):
        """Test with XOR-like problem (not linearly separable)."""
        # XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        data = np.array([
            [0.0, 0.0, 0],
            [0.0, 1.0, 1],
            [1.0, 0.0, 1],
            [1.0, 1.0, 0]
        ])
        
        # No single split should perfectly separate XOR
        gain_f0 = gain(data, feature_index=0, threshold=0.5)
        gain_f1 = gain(data, feature_index=1, threshold=0.5)
        
        # Both gains should be 0 (no improvement possible with single split)
        self.assertAlmostEqual(gain_f0, 0.0, places=5,
                              msg="Single split can't solve XOR")
        self.assertAlmostEqual(gain_f1, 0.0, places=5,
                              msg="Single split can't solve XOR")


if __name__ == '__main__':
    unittest.main(verbosity=2)
