"""Fast tests for the final leakage-safe regression helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.common.regression import (
    NonEmptyLassoSelector,
    bootstrap_draws,
    paired_mae_bootstrap,
    strategy_pipeline_and_grid,
)


class RegressionTests(unittest.TestCase):
    def test_nonempty_lasso_selector_always_keeps_a_feature(self):
        x = np.zeros((20, 5))
        y = np.linspace(0, 1, 20)
        selector = NonEmptyLassoSelector(alpha=100.0).fit(x, y)
        self.assertEqual(selector.transform(x).shape, (20, 1))
        self.assertTrue(selector.fallback_)

    def test_ablation_pipeline_fits_fold_local_transformers(self):
        random = np.random.default_rng(7)
        x = random.normal(size=(30, 10))
        y = 2 * x[:, 0] + random.normal(size=30)
        pipeline, _ = strategy_pipeline_and_grid("lasso", radiomics_count=4)
        pipeline.set_params(
            model__alpha=1.0,
            preprocess__radiomics__selector__alpha=0.1,
        )
        pipeline.fit(x[:24], y[:24])
        predictions = pipeline.predict(x[24:])
        self.assertEqual(predictions.shape, (6,))

    def test_bootstrap_draws_are_reproducible_and_have_oob_patients(self):
        first = bootstrap_draws(148, 42, 4)
        second = bootstrap_draws(148, 42, 4)
        for replicate in first:
            np.testing.assert_array_equal(first[replicate][0], second[replicate][0])
            np.testing.assert_array_equal(first[replicate][1], second[replicate][1])
            self.assertGreaterEqual(len(first[replicate][1]), 20)

    def test_paired_bootstrap_preserves_prediction_pairing(self):
        observed = np.arange(12, dtype=float)
        perfect = observed.copy()
        shifted = observed + 2.0
        intervals, deltas = paired_mae_bootstrap(
            observed, {"perfect": perfect, "shifted": shifted}, 500, 42
        )
        self.assertEqual(float(intervals.loc[0, "mae"]), 0.0)
        self.assertEqual(
            float(deltas.loc[0, "delta_mae_candidate_minus_reference"]), 2.0
        )
        self.assertEqual(float(deltas.loc[0, "probability_candidate_lower_mae"]), 0.0)


if __name__ == "__main__":
    unittest.main()
