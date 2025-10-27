"""
Comprehensive tests for information gain calculation.

Tests cover:
- Perfect splits (gain = initial entropy)
- No-improvement splits (gain = 0)
- Various split qualities
- Edge cases (all samples on one side)
- Multi-class scenarios
- Numerical precision and properties
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import TreeUtils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TreeUtils import entropy, gain


class TestInformationGain(unittest.TestCase):
    """Test suite for information gain calculation."""

    def assertAlmostEqualGain(self, actual, expected, places=7, msg=None):
        """Helper to compare gain values with tolerance."""
        self.assertAlmostEqual(actual, expected, places=places, msg=msg)

    # ========== Perfect Split Tests ==========

    def test_perfect_binary_split(self):
        """Perfect split that completely separates classes should give maximum gain."""
        # All class 0 on left (feature <= 1.5), all class 1 on right (feature > 1.5)
        data = np.array([
            [1.0, 5.0, 0],
            [1.0, 6.0, 0],
            [2.0, 5.0, 1],
            [2.0, 6.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=1.5)
        # Initial entropy = 1.0 (balanced binary)
        # After split: both subsets are pure (entropy = 0)
        # Gain = 1.0 - 0 = 1.0
        self.assertAlmostEqualGain(result, 1.0, msg="Perfect split should have gain equal to initial entropy")

    def test_perfect_split_three_classes(self):
        """Perfect split separating three classes."""
        data = np.array([
            [1.0, 5.0, 0],
            [1.0, 6.0, 0],
            [2.0, 5.0, 1],
            [2.0, 6.0, 1],
            [3.0, 5.0, 2],
            [3.0, 6.0, 2]
        ])
        result = gain(data, feature_index=0, threshold=1.5)
        # Left subset: all class 0 (pure)
        # Right subset: classes 1 and 2 (balanced binary)
        initial_entropy = entropy(data)
        left_entropy = 0.0  # Pure
        right_entropy = 1.0  # Balanced binary between classes 1 and 2
        expected_gain = initial_entropy - (2/6 * left_entropy + 4/6 * right_entropy)
        self.assertAlmostEqualGain(result, expected_gain, 
                                  msg="Split separating one pure class should have correct gain")

    # ========== No Information Gain Tests ==========

    def test_no_gain_useless_split(self):
        """Split that doesn't change class distribution should have gain ≈ 0."""
        # Both sides of split have same class distribution
        data = np.array([
            [1.0, 5.0, 0],
            [1.0, 6.0, 1],
            [2.0, 5.0, 0],
            [2.0, 6.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=1.5)
        # Both subsets have same 50-50 distribution as original
        # Gain should be approximately 0
        self.assertAlmostEqualGain(result, 0.0, places=5,
                                  msg="Useless split should have gain close to 0")

    def test_no_gain_pure_dataset(self):
        """Splitting an already pure dataset should give gain = 0."""
        data = np.array([
            [1.0, 5.0, 0],
            [1.5, 6.0, 0],
            [2.0, 7.0, 0],
            [2.5, 8.0, 0]
        ])
        result = gain(data, feature_index=0, threshold=1.75)
        # Initial entropy = 0, final entropy = 0
        # Gain = 0
        self.assertAlmostEqualGain(result, 0.0, msg="Splitting pure dataset should give gain of 0")

    # ========== Partial Improvement Tests ==========

    def test_partial_improvement_split(self):
        """Split that partially improves purity."""
        # 75% class 0, 25% class 1
        # Split moves towards more purity but not perfect
        data = np.array([
            [1.0, 5.0, 0],
            [1.2, 5.5, 0],
            [1.4, 6.0, 0],
            [2.0, 5.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=1.5)
        
        # Calculate expected gain manually
        initial_entropy = entropy(data)
        left_data = data[data[:, 0] <= 1.5]
        right_data = data[data[:, 0] > 1.5]
        
        left_entropy = entropy(left_data)
        right_entropy = entropy(right_data)
        
        weighted_entropy = (len(left_data) / len(data)) * left_entropy + \
                          (len(right_data) / len(data)) * right_entropy
        expected_gain = initial_entropy - weighted_entropy
        
        self.assertAlmostEqualGain(result, expected_gain, places=5,
                                  msg="Partial improvement split gain mismatch")
        self.assertGreater(result, 0.0, msg="Partial improvement should have positive gain")
        # Allow for small floating point errors - gain should be <= initial entropy
        self.assertLessEqual(result, initial_entropy + 1e-8, 
                       msg="Partial improvement gain should be less than or equal to initial entropy")

    def test_varying_split_quality(self):
        """Test that better splits have higher gains."""
        # Create dataset where one feature perfectly separates, another doesn't
        data = np.array([
            [1.0, 10.0, 0],
            [1.0, 20.0, 0],
            [2.0, 10.0, 1],
            [2.0, 20.0, 1]
        ])
        
        # Perfect split on feature 0
        gain_perfect = gain(data, feature_index=0, threshold=1.5)
        
        # Useless split on feature 1
        gain_useless = gain(data, feature_index=1, threshold=15.0)
        
        self.assertGreater(gain_perfect, gain_useless, 
                          msg="Perfect split should have higher gain than useless split")

    # ========== Edge Case Tests ==========

    def test_all_samples_left_side(self):
        """Threshold puts all samples on left side."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=10.0)
        # All samples on left, right is empty
        # This should give gain = 0 (no improvement)
        self.assertAlmostEqualGain(result, 0.0, msg="All samples on one side should give gain of 0")

    def test_all_samples_right_side(self):
        """Threshold puts all samples on right side."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=-10.0)
        # All samples on right, left is empty
        # This should give gain = 0 (no improvement)
        self.assertAlmostEqualGain(result, 0.0, msg="All samples on one side should give gain of 0")

    def test_single_sample_left(self):
        """Split with only one sample on left side."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [2.5, 7.0, 1],
            [3.0, 8.0, 1]
        ])
        result = gain(data, feature_index=0, threshold=1.5)
        
        # Calculate expected
        initial_entropy = entropy(data)
        left_entropy = 0.0  # Single sample is pure
        right_data = data[data[:, 0] > 1.5]
        right_entropy = entropy(right_data)
        
        expected_gain = initial_entropy - (1/4 * left_entropy + 3/4 * right_entropy)
        
        self.assertAlmostEqualGain(result, expected_gain, 
                                  msg="Single sample split gain mismatch")

    # ========== Different Threshold Tests ==========

    def test_multiple_thresholds_same_feature(self):
        """Test different thresholds on the same feature."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        
        # Try different thresholds
        gain_1 = gain(data, feature_index=0, threshold=1.5)
        gain_2 = gain(data, feature_index=0, threshold=2.5)
        gain_3 = gain(data, feature_index=0, threshold=3.5)
        
        # The middle threshold should give the best gain (perfect split)
        self.assertGreater(gain_2, gain_1, msg="Better threshold should have higher gain")
        self.assertGreater(gain_2, gain_3, msg="Better threshold should have higher gain")

    def test_threshold_at_boundary(self):
        """Test threshold exactly at a data point."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [2.0, 7.0, 1],  # Note: same value as previous row
            [3.0, 8.0, 1]
        ])
        
        # Threshold at 2.0 - should use <= for left, > for right
        result = gain(data, feature_index=0, threshold=2.0)
        
        # Left: values <= 2.0 (rows 0, 1, 2)
        # Right: values > 2.0 (row 3)
        left_data = data[data[:, 0] <= 2.0]
        right_data = data[data[:, 0] > 2.0]
        
        self.assertEqual(len(left_data), 3, msg="Left split should have 3 samples")
        self.assertEqual(len(right_data), 1, msg="Right split should have 1 sample")

    # ========== Multi-feature Tests ==========

    def test_different_features(self):
        """Test gain calculation on different features."""
        data = np.array([
            [1.0, 10.0, 0],
            [1.0, 20.0, 0],
            [2.0, 10.0, 1],
            [2.0, 20.0, 1]
        ])
        
        # Feature 0 perfectly separates classes
        gain_feature_0 = gain(data, feature_index=0, threshold=1.5)
        
        # Feature 1 doesn't separate classes well
        gain_feature_1 = gain(data, feature_index=1, threshold=15.0)
        
        self.assertGreater(gain_feature_0, 0.9, 
                          msg="Perfect split should have high gain")
        self.assertLess(gain_feature_1, 0.1, 
                       msg="Poor split should have low gain")

    # ========== Multi-class Tests ==========

    def test_three_class_split(self):
        """Test gain calculation with three classes."""
        data = np.array([
            [1.0, 5.0, 0],
            [1.5, 5.5, 0],
            [2.0, 6.0, 1],
            [2.5, 6.5, 1],
            [3.0, 7.0, 2],
            [3.5, 7.5, 2]
        ])
        
        # Split that separates class 0 from classes 1 and 2
        result = gain(data, feature_index=0, threshold=1.75)
        
        initial_entropy = entropy(data)
        left_data = data[data[:, 0] <= 1.75]
        right_data = data[data[:, 0] > 1.75]
        
        left_entropy = entropy(left_data)
        right_entropy = entropy(right_data)
        
        expected_gain = initial_entropy - \
                       (len(left_data)/len(data) * left_entropy + 
                        len(right_data)/len(data) * right_entropy)
        
        self.assertAlmostEqualGain(result, expected_gain, 
                                  msg="Three-class split gain mismatch")

    # ========== Property Tests ==========

    def test_gain_non_negative(self):
        """Information gain should always be non-negative."""
        test_cases = [
            (np.array([[i, i, i % 2] for i in range(10)]), 0, 5.0),
            (np.array([[i, i, i % 3] for i in range(15)]), 0, 7.5),
            (np.array([[i, 2*i, i % 2] for i in range(20)]), 1, 20.0),
        ]
        
        for data, feature_idx, threshold in test_cases:
            result = gain(data, feature_idx, threshold)
            self.assertGreaterEqual(result, 0.0, 
                                   msg="Information gain should be non-negative")

    def test_gain_bounded_by_entropy(self):
        """Information gain should not exceed the initial entropy."""
        test_cases = [
            (np.array([[i, i, i % 2] for i in range(10)]), 0, 5.0),
            (np.array([[i, i, i % 3] for i in range(15)]), 0, 7.5),
            (np.array([[i, 2*i, i % 2] for i in range(20)]), 1, 20.0),
        ]
        
        for data, feature_idx, threshold in test_cases:
            initial_entropy = entropy(data)
            result = gain(data, feature_idx, threshold)
            self.assertLessEqual(result, initial_entropy + 1e-9,  # Small tolerance for floating point
                               msg="Gain should not exceed initial entropy")

    def test_gain_consistency(self):
        """Same split should give same gain when recalculated."""
        data = np.array([
            [1.0, 5.0, 0],
            [2.0, 6.0, 0],
            [3.0, 7.0, 1],
            [4.0, 8.0, 1]
        ])
        
        gain_1 = gain(data, feature_index=0, threshold=2.5)
        gain_2 = gain(data, feature_index=0, threshold=2.5)
        
        self.assertEqual(gain_1, gain_2, msg="Gain calculation should be deterministic")

    # ========== Large Dataset Tests ==========

    def test_large_dataset_gain(self):
        """Test gain calculation on a larger dataset."""
        np.random.seed(42)
        # Create a dataset where feature 0 correlates with class
        n_samples = 1000
        feature_0 = np.random.randn(n_samples)
        feature_1 = np.random.randn(n_samples)
        labels = (feature_0 > 0).astype(int)
        
        data = np.column_stack([feature_0, feature_1, labels])
        
        # Split on feature 0 at 0 should give good gain
        result = gain(data, feature_index=0, threshold=0.0)
        
        self.assertGreater(result, 0.1, 
                          msg="Correlated feature should provide positive gain")

    # ========== Numerical Stability Tests ==========

    def test_no_nan_or_inf(self):
        """Ensure gain calculation doesn't produce NaN or Inf."""
        test_cases = [
            np.array([[1.0, 2.0, 0], [1.0, 2.0, 0]]),  # All same
            np.array([[i, i, i % 2] for i in range(100)]),  # Large
            np.array([[i, i, 0] for i in range(5)]),  # Pure
        ]
        
        for data in test_cases:
            result = gain(data, feature_index=0, threshold=data[:, 0].mean())
            self.assertFalse(np.isnan(result), msg="Gain should not be NaN")
            self.assertFalse(np.isinf(result), msg="Gain should not be infinite")


if __name__ == '__main__':
    unittest.main(verbosity=2)
