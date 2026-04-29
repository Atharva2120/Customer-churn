import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--target", default=None, help="Target column name to drop if present")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

    out = df.copy()
    out["churn_prediction"] = preds
    if proba is not None:
        out["churn_probability"] = proba

    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
