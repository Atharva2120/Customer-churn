import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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


def get_feature_names(preprocess: ColumnTransformer) -> list:
    feature_names = []
    for name, transformer, cols in preprocess.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if isinstance(transformer, Pipeline):
            last_step = transformer.steps[-1][1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
        else:
            names = cols

        feature_names.extend(list(names))
    return feature_names


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="binary"),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    else:
        metrics["roc_auc"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
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
        "logistic_regression": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        score = metrics.get("roc_auc") or metrics["f1"]
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pkl")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    importance_path = os.path.join(args.output_dir, "feature_importance.csv")

    joblib.dump(best_pipeline, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # Feature importance
    model = best_pipeline.named_steps["model"]
    preprocess = best_pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(preprocess)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(importance_path, index=False)

    print(f"Best model: {best_name}")
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    if importances is not None:
        print(f"Saved feature importance to {importance_path}")


if __name__ == "__main__":
    main()
