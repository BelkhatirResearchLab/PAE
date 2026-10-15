"""Fast tests for shared identifiers, metrics and selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from src.common.feature_processing import SafeL1LogisticSelector, SafeLassoSelector
from src.common.io_utils import normalise_patient_id, require_unique_ids
from src.common.metrics import choose_youden_threshold, classification_metrics


class CommonTests(unittest.TestCase):
    def test_patient_ids_are_normalised(self) -> None:
        self.assertEqual(normalise_patient_id("PAE-001"), "PAE1")
        self.assertEqual(normalise_patient_id(" 12 "), "PAE12")

    def test_duplicate_ids_are_rejected(self) -> None:
        table = pd.DataFrame({"patient_id": ["PAE1", "PAE1"]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            require_unique_ids(table, "test")

    def test_youden_threshold_uses_training_scores(self) -> None:
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        threshold = choose_youden_threshold(y, scores)
        result = classification_metrics(y, scores, threshold)
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertEqual(result["specificity"], 1.0)

    def test_safe_lasso_selector_keeps_one_component(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(36, 8))
        y = 2.0 * x[:, 0] + rng.normal(scale=0.2, size=36)
        selector = SafeLassoSelector(cv_splits=3, n_jobs=1).fit(x, y)
        transformed = selector.transform(x)
        self.assertEqual(transformed.shape[0], len(x))
        self.assertTrue(1 <= transformed.shape[1] <= x.shape[1])
        self.assertGreater(selector.selected_alpha_, 0)

    def test_safe_l1_selector_keeps_one_component(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(48, 7))
        y = (x[:, 0] + 0.3 * rng.normal(size=48) > 0).astype(int)
        selector = SafeL1LogisticSelector(cv_splits=3, n_jobs=1).fit(x, y)
        transformed = selector.transform(x)
        self.assertEqual(transformed.shape[0], len(x))
        self.assertTrue(1 <= transformed.shape[1] <= x.shape[1])
        self.assertGreater(selector.selected_c_, 0)
