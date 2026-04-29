import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.columns.difference(cat_cols)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="binary"),
    }
    metrics["roc_auc"] = roc_auc_score(y_test, proba) if proba is not None else None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")

    df = df.dropna(subset=[args.target])
    X = df.drop(columns=[args.target])
    y = df[args.target]

    if y.dtype == "object":
        y = y.astype(str).str.lower().map({"yes": 1, "true": 1, "churn": 1, "1": 1, "no": 0, "false": 0, "not churn": 0, "0": 0})
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = build_preprocessor(X)

    models = {
        "random_forest": (RandomForestClassifier(random_state=42), {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_split": [2, 5, 10],
        }),
        "gradient_boosting": (GradientBoostingClassifier(random_state=42), {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        }),
    }

    results = {}
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, (clf, params) in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            n_iter=12,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        metrics = evaluate(tuned, X_test, y_test)
        results[name] = {"best_params": search.best_params_, "metrics": metrics}

        score = metrics.get("roc_auc") or metrics["f1"]
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = tuned

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "tuned_model.pkl")
    metrics_path = os.path.join(args.output_dir, "tuned_metrics.json")

    joblib.dump(best_pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Best tuned model: {best_name}")
    print(f"Saved tuned model to {model_path}")
    print(f"Saved tuned metrics to {metrics_path}")


if __name__ == "__main__":
    main()
