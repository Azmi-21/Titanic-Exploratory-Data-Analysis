import pandas as pd
from sklearn.model_selection import train_test_split
import os

from eda_titanic import DATA_PATH

# Q1-b: train/test split

def main():
    # Load the Titanic dataset
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic.csv')
    df = pd.read_csv(DATA_PATH)

    # Select features and target
    target = "Survived"
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    X = df[features]
    y = df[target]

    # Split training and test sets 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=433,   
        stratify=y          # Preserve ratios
    )

    # Print basic info
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

if __name__ == "__main__":
    main()
