import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


DATA_PATH = "data/online_shopping_purchase_prediction.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def build_model():

    categorical_features = [
        "Gender",
        "DiscountUsed",
        "DeviceType",
        "Membership"
    ]

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "WebsiteVisits",
        "PagesViewed",
        "TimeSpentMinutes",
        "PreviousPurchases",
        "CartItems",
        "CustomerRating",
        "PurchaseAmount"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    return model
