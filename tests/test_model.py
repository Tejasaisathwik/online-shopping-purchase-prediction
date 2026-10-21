import os
import unittest
import pandas as pd

from src.model import load_data, build_model


class TestOnlineShoppingDataset(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists(
                "data/online_shopping_purchase_prediction.csv"
            )
        )

    def test_dataset_not_empty(self):
        df = load_data()
        self.assertGreater(len(df), 0)

    def test_target_column_exists(self):
        df = load_data()
        self.assertIn("Purchased", df.columns)

    def test_number_of_columns(self):
        df = load_data()
        self.assertEqual(len(df.columns), 14)

    def test_target_values(self):
        df = load_data()

        self.assertTrue(
            set(df["Purchased"].dropna().unique()).issubset({0, 1})
        )

    def test_model_can_be_created(self):
        model = build_model()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
