import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output_dir", default=os.path.join("outputs", "eda"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.data)

    # Target distribution
    if args.target in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[args.target])
        plt.title("Target Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "target_distribution.png"))
        plt.close()

    # Numeric histograms
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(figsize=(12, 8), bins=20)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "numeric_histograms.png"))
        plt.close()

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "correlation_heatmap.png"))
        plt.close()

    print(f"EDA plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
