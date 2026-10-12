from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.model import load_data, build_model


def main():

    # Load dataset
    df = load_data()

    # Separate features and target
    X = df.drop(columns=["Purchased"])
    y = df["Purchased"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create model
    model = build_model()

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Training Completed")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "model.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
