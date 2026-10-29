import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "healthy")

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        content = response.text
        self.assertIn("online_shopping_api_model_loaded", content)
        self.assertIn("student_api_model_loaded", content)

    def test_predict_endpoint_success(self):
        payload = {
            "Age": 46,
            "Gender": "Male",
            "MonthlyIncome": 141446.0,
            "WebsiteVisits": 12,
            "PagesViewed": 11,
            "TimeSpentMinutes": 8,
            "PreviousPurchases": 7,
            "CartItems": 3,
            "DiscountUsed": "No",
            "DeviceType": "Mobile",
            "Membership": "Basic",
            "CustomerRating": 4.8,
            "PurchaseAmount": 9154.39,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn(data["prediction"], [0, 1])

    def test_predict_validation_error(self):
        # Missing required fields
        response = self.client.post("/predict", json={"Age": 46})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
