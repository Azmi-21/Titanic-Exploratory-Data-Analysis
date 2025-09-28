import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    # Load data
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic.csv')
    df = pd.read_csv(DATA_PATH)

    # Select features and target
    target = "Survived"
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    # Take an explicit copy to avoid SettingWithCopy warnings
    X = df.loc[:, features].copy()
    y = df[target]

    # Handle missing values (Age: median, Embarked: mode)
    # Avoid chained assignment and inplace operations on a Series
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=433, stratify=y
    )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Logistic Regression Results")
    print("---------------------------")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")

    # Classification reports
    print("Training Classification Report:")
    print(classification_report(y_train, y_train_pred))

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    main()
