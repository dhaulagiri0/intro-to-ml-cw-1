"""
Comprehensive tests for entropy calculation.

Tests cover:
- Pure datasets (entropy = 0)
- Balanced datasets (maximum entropy)
- Imbalanced datasets
- Binary and multi-class scenarios
- Edge cases (single sample, empty subsets)
- Numerical precision
"""

import os
import sys
import unittest

import numpy as np

# Add parent directory to path to import TreeUtils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TreeUtils import entropy


class TestEntropyCalculation(unittest.TestCase):
    """Test suite for entropy calculation."""

    def assertAlmostEqualEntropy(self, actual, expected, places=7, msg=None):
        """Helper to compare entropy values with tolerance."""
        self.assertAlmostEqual(actual, expected, places=places, msg=msg)

    # ========== Pure Dataset Tests ==========

    def test_pure_dataset_single_class(self):
        """Entropy of a pure dataset (all same class) should be 0."""
        data = np.array([[1.0, 2.0, 0], [1.5, 2.5, 0], [2.0, 3.0, 0], [2.5, 3.5, 0]])
        result = entropy(data)
        self.assertAlmostEqualEntropy(result, 0.0, msg="Pure dataset should have entropy of 0")

    def test_pure_dataset_single_sample(self):
        """Single sample should have entropy of 0."""
        data = np.array([[1.0, 2.0, 1]])
        result = entropy(data)
        self.assertAlmostEqualEntropy(result, 0.0, msg="Single sample should have entropy of 0")

    # ========== Binary Classification Tests ==========

    def test_binary_balanced_dataset(self):
        """Perfectly balanced binary dataset should have entropy = 1."""
        data = np.array([[1.0, 2.0, 0], [1.5, 2.5, 0], [2.0, 3.0, 1], [2.5, 3.5, 1]])
        result = entropy(data)
        # For binary: H = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
        self.assertAlmostEqualEntropy(result, 1.0, msg="Balanced binary dataset should have entropy of 1")

    def test_binary_imbalanced_75_25(self):
        """Binary dataset with 75-25 split."""
        data = np.array([[1.0, 2.0, 0], [1.5, 2.5, 0], [2.0, 3.0, 0], [2.5, 3.5, 1]])
        result = entropy(data)
        # H = -0.75*log2(0.75) - 0.25*log2(0.25)
        # H = -0.75*(-0.415) - 0.25*(-2.0)
        # H ≈ 0.311 + 0.5 = 0.811
        expected = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
        self.assertAlmostEqualEntropy(result, expected, msg="75-25 binary split entropy mismatch")

    def test_binary_imbalanced_90_10(self):
        """Highly imbalanced binary dataset (90-10 split)."""
        data = np.array([[i, i, 0] for i in range(9)] + [[9, 9, 1]])
        result = entropy(data)
        # H = -0.9*log2(0.9) - 0.1*log2(0.1)
        expected = -0.9 * np.log2(0.9) - 0.1 * np.log2(0.1)
        self.assertAlmostEqualEntropy(result, expected, msg="90-10 binary split entropy mismatch")

    def test_binary_extreme_imbalance_99_1(self):
        """Extremely imbalanced dataset (99-1 split)."""
        data = np.array([[i, i, 0] for i in range(99)] + [[99, 99, 1]])
        result = entropy(data)
        expected = -0.99 * np.log2(0.99) - 0.01 * np.log2(0.01)
        self.assertAlmostEqualEntropy(result, expected, places=5, msg="99-1 binary split entropy mismatch")

    # ========== Multi-class Tests ==========

    def test_three_class_balanced(self):
        """Three equally distributed classes."""
        data = np.array([[1.0, 2.0, 0], [1.5, 2.5, 0], [2.0, 3.0, 1], [2.5, 3.5, 1], [3.0, 4.0, 2], [3.5, 4.5, 2]])
        result = entropy(data)
        # H = -3 * (1/3 * log2(1/3)) = -log2(1/3) = log2(3)
        expected = np.log2(3)
        self.assertAlmostEqualEntropy(result, expected, msg="Balanced 3-class entropy mismatch")

    def test_four_class_balanced(self):
        """Four equally distributed classes."""
        data = np.array([[i, i, i % 4] for i in range(40)])
        result = entropy(data)
        # H = -4 * (1/4 * log2(1/4)) = -log2(1/4) = log2(4) = 2.0
        expected = 2.0
        self.assertAlmostEqualEntropy(result, expected, msg="Balanced 4-class entropy should be 2.0")

    def test_three_class_imbalanced(self):
        """Three classes with imbalanced distribution (50-30-20)."""
        data = np.array(
            [[i, i, 0] for i in range(50)] + [[i, i, 1] for i in range(30)] + [[i, i, 2] for i in range(20)])
        result = entropy(data)
        # H = -0.5*log2(0.5) - 0.3*log2(0.3) - 0.2*log2(0.2)
        expected = -0.5 * np.log2(0.5) - 0.3 * np.log2(0.3) - 0.2 * np.log2(0.2)
        self.assertAlmostEqualEntropy(result, expected, msg="Imbalanced 3-class entropy mismatch")

    def test_five_class_diverse(self):
        """Five classes with varying distributions."""
        # Distribution: 40-25-20-10-5 (percentages)
        data = np.array(
            [[i, i, 0] for i in range(40)] + [[i, i, 1] for i in range(25)] + [[i, i, 2] for i in range(20)] + [
                [i, i, 3] for i in range(10)] + [[i, i, 4] for i in range(5)])
        result = entropy(data)
        probs = np.array([40, 25, 20, 10, 5]) / 100.0
        expected = -np.sum(probs * np.log2(probs))
        self.assertAlmostEqualEntropy(result, expected, msg="5-class diverse entropy mismatch")

    # ========== Edge Cases ==========

    def test_very_large_dataset(self):
        """Test entropy calculation on a large dataset."""
        # 10000 samples, balanced binary
        data = np.array([[i, i, i % 2] for i in range(10000)])
        result = entropy(data)
        self.assertAlmostEqualEntropy(result, 1.0, msg="Large balanced dataset should have entropy of 1")

    def test_float_labels(self):
        """Test with float labels (should still work)."""
        data = np.array([[1.0, 2.0, 0.0], [1.5, 2.5, 0.0], [2.0, 3.0, 1.0], [2.5, 3.5, 1.0]])
        result = entropy(data)
        self.assertAlmostEqualEntropy(result, 1.0, msg="Float labels should work correctly")

    def test_negative_labels(self):
        """Test with negative class labels."""
        data = np.array([[1.0, 2.0, -1], [1.5, 2.5, -1], [2.0, 3.0, 1], [2.5, 3.5, 1]])
        result = entropy(data)
        self.assertAlmostEqualEntropy(result, 1.0, msg="Negative labels should work correctly")

    # ========== Entropy Properties Tests ==========

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        test_cases = [np.array([[i, i, i % 2] for i in range(10)]), np.array([[i, i, i % 3] for i in range(15)]),
            np.array([[i, i, 0] for i in range(10)]), ]
        for data in test_cases:
            result = entropy(data)
            # Allow small negative values due to floating point arithmetic with the 1e-9 offset
            self.assertGreaterEqual(result, -1e-8,
                                    msg="Entropy should be non-negative (allowing floating point tolerance)")

    def test_entropy_bounded_binary(self):
        """For binary classification, entropy should be between 0 and 1."""
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n_class_0 = int(100 * ratio)
            n_class_1 = 100 - n_class_0
            data = np.array([[i, i, 0] for i in range(n_class_0)] + [[i, i, 1] for i in range(n_class_1)])
            result = entropy(data)
            self.assertGreaterEqual(result, 0.0, msg=f"Entropy should be >= 0 for ratio {ratio}")
            self.assertLessEqual(result, 1.0, msg=f"Binary entropy should be <= 1 for ratio {ratio}")

    def test_entropy_bounded_multiclass(self):
        """For k classes, entropy should be between 0 and log2(k)."""
        k = 5
        data = np.array([[i, i, i % k] for i in range(100)])
        result = entropy(data)
        max_entropy = np.log2(k)
        self.assertGreaterEqual(result, 0.0, msg="Entropy should be >= 0")
        self.assertLessEqual(result, max_entropy, msg=f"Entropy should be <= log2({k})")

    def test_entropy_increases_with_balance(self):
        """Entropy should increase as distribution becomes more balanced."""
        # Test increasingly balanced distributions
        distributions = [np.array([[0, 0, 0] for _ in range(90)] + [[1, 1, 1] for _ in range(10)]),  # 90-10
            np.array([[0, 0, 0] for _ in range(70)] + [[1, 1, 1] for _ in range(30)]),  # 70-30
            np.array([[0, 0, 0] for _ in range(50)] + [[1, 1, 1] for _ in range(50)]),  # 50-50
        ]
        entropies = [entropy(data) for data in distributions]
        # Each entropy should be greater than or equal to the previous
        for i in range(len(entropies) - 1):
            self.assertLess(entropies[i], entropies[i + 1],
                            msg="Entropy should increase as distribution becomes more balanced")

    # ========== Numerical Stability Tests ==========

    def test_no_log_zero_error(self):
        """Ensure no log(0) errors occur."""
        # This should not raise any warnings or errors
        data = np.array([[i, i, 0] for i in range(100)])
        try:
            result = entropy(data)
            self.assertIsInstance(result, (float, np.floating), msg="Entropy should return a numeric value")
            self.assertFalse(np.isnan(result), msg="Entropy should not be NaN")
            self.assertFalse(np.isinf(result), msg="Entropy should not be infinite")
        except Exception as e:
            self.fail(f"Entropy calculation raised exception: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
