import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay


# Data paths  & constants
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Titanic.csv')
RESULTS_DIR = "results"
MODELS_DIR = "models"
FIGURES_DIR = "figures"

RANDOM_STATE = 433
TEST_SIZE = 0.2

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def build_preprocessing_pipeline():

    # Which columns to treat as numeric vs categorical
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    categorical_cols = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    return preprocessor, numeric_cols, categorical_cols

def main():
    # Load data
    df = load_data()
    print("Loaded dataset shape:", df.shape)

    # Feature selection & target
    target_col = "Survived"
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols]
    y = df[target_col]

    # Train/test split (same as logistic regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train / Test sizes: {X_train.shape[0]} / {X_test.shape[0]}")

    # Build preprocessing + model pipeline
    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline()
    perceptron = Perceptron(random_state=RANDOM_STATE, max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preproc", preprocessor),
        ("clf", perceptron)
    ])

    # Hyperparameter grid (small & practical)
    param_grid = {
        "clf__penalty": [None, "l2", "l1", "elasticnet"],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__max_iter": [1000, 3000],
        "clf__class_weight": [None, "balanced"],
        "clf__eta0": [1.0, 0.1]   # initial learning rate parameter
    }

    # Use StratifiedKFold for stable CV on imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    print("Fitting GridSearchCV on training data...")
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    best_model = grid.best_estimator_

    # Evaluate on train and test sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_report = classification_report(y_train, y_train_pred, digits=4)
    test_report = classification_report(y_test, y_test_pred, digits=4)

    print("\n=== Final evaluation ===")
    print(f"Training accuracy: {train_acc:.4f}")
    print(train_report)
    print(f"Test accuracy: {test_acc:.4f}")
    print(test_report)

    # Save results to text files
    with open(os.path.join(RESULTS_DIR, "perceptron_best_params.txt"), "w") as f:
        f.write(str(grid.best_params_) + "\n")
        f.write(f"best_cv_accuracy: {grid.best_score_}\n")

    with open(os.path.join(RESULTS_DIR, "perceptron_train_report.txt"), "w") as f:
        f.write(f"Training accuracy: {train_acc:.4f}\n\n")
        f.write(train_report)

    with open(os.path.join(RESULTS_DIR, "perceptron_test_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(test_report)

    # Save confusion matrix figures
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=ax1)
    ax1.set_title("Confusion matrix (train)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, "confusion_train_perceptron.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4,4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax2)
    ax2.set_title("Confusion matrix (test)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "confusion_test_perceptron.png"))
    plt.close(fig2)

    # Save best model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "perceptron_best.pkl"))
    print("Saved model and results. Done.")

if __name__ == "__main__":
    main()
