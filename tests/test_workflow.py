"""Checks for mistakes that would invalidate the evaluation."""
import unittest
import numpy as np
import pandas as pd
from src.churn import FEATURES, NUMERIC, preprocessor, split_data, choose_threshold


class WorkflowTests(unittest.TestCase):
    def test_outcome_fields_are_not_features(self):
        excluded = {"Churn Value", "Churn Label", "Churn Score", "Churn Reason", "CLTV", "CustomerID", "Zip Code"}
        self.assertFalse(set(FEATURES) & excluded)
        self.assertEqual(len(FEATURES), len(set(FEATURES)))

    def test_splits_are_disjoint_and_complete(self):
        data = pd.DataFrame({c: np.arange(100) for c in FEATURES})
        data["Churn Value"] = [0, 0, 0, 1] * 25
        a, b, c, *_ = split_data(data)
        sets = [set(x.index) for x in [a, b, c]]
        self.assertEqual([len(x) for x in sets], [60, 20, 20])
        self.assertEqual(len(set.union(*sets)), 100)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_preprocessor_uses_training_median_and_accepts_unseen_category(self):
        train = pd.DataFrame({c: ["No", "Yes", "No"] for c in FEATURES})
        for c in NUMERIC:
            train[c] = [1., 3., np.nan]
        processor = preprocessor().fit(train)
        unseen = train.iloc[[0]].copy()
        unseen["Contract"] = "Previously unseen plan"
        unseen["Total Charges"] = np.nan
        transformed = processor.transform(unseen)
        self.assertTrue(np.isfinite(transformed).all())
        imputer = processor.named_transformers_["numeric"].named_steps["impute"]
        np.testing.assert_array_equal(imputer.statistics_, [2., 2., 2.])

    def test_threshold_optimizes_validation_f2(self):
        threshold, table = choose_threshold(np.array([0, 0, 1, 1]), np.array([.1, .4, .3, .8]))
        selected = table.loc[np.isclose(table.threshold, threshold)].iloc[0]
        self.assertAlmostEqual(selected.f2, table.f2.max())


if __name__ == "__main__":
    unittest.main()
