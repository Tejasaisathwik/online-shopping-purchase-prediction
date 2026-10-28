import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_model, load_data


class TestOnlineShoppingDataset(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists(
                "data/online_shopping_purchase_prediction.csv"
            )
            or os.path.exists("online_shopping.csv")
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
