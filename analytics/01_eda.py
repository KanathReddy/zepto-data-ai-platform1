from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analytics" / "titanic.csv"
OUTPUT_DIR = ROOT / "analytics" / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_dataset() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    df.to_csv(DATA_PATH, index=False)
    return df


def summarize_missing(df: pd.DataFrame) -> None:
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("Missing value rates:")
    print(missing)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["embarked"]).copy()
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
    cleaned["fare"] = cleaned["fare"].fillna(cleaned["fare"].median())
    cleaned = cleaned.drop(columns=["deck"], errors="ignore")
    return cleaned


def plot_univariate(cleaned: pd.DataFrame) -> None:
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(cleaned[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=cleaned[col], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{col}_distribution.png")
        plt.close(fig)


def outlier_counts(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count = ((df[col] < low) | (df[col] > high)).sum()
        result.append({"column": col, "q1": q1, "q3": q3, "iqr": iqr, "low": low, "high": high, "outliers": int(count)})
    return pd.DataFrame(result)


def bivariate_summary(df: pd.DataFrame) -> None:
    surv_sex = df.groupby("sex")["survived"].mean().sort_values(ascending=False)
    surv_class = df.groupby("pclass")["survived"].mean().sort_values(ascending=False)
    surv_sex_class = df.groupby(["sex", "pclass"])["survived"].mean().unstack()
    print("Survival by sex:\n", surv_sex)
    print("\nSurvival by class:\n", surv_class)
    print("\nSurvival by sex and class:\n", surv_sex_class)
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    print("\nCorrelation matrix:\n", corr)


def standardize_check(df: pd.DataFrame) -> None:
    std_age = (df["age"] - df["age"].mean()) / df["age"].std(ddof=0)
    std_fare = (df["fare"] - df["fare"].mean()) / df["fare"].std(ddof=0)
    print("Age z-score mean/std:", std_age.mean(), std_age.std(ddof=0))
    print("Fare z-score mean/std:", std_fare.mean(), std_fare.std(ddof=0))


def main() -> None:
    df = load_dataset()
    print(df.info())
    print(df.describe())
    print(df.shape)
    summarize_missing(df)
    cleaned = clean_data(df)
    cleaned.to_csv(DATA_PATH, index=False)
    plot_univariate(cleaned)
    print(outlier_counts(cleaned))
    print("Fare mean, median, mode:", cleaned["fare"].mean(), cleaned["fare"].median(), cleaned["fare"].mode().iloc[0])
    bivariate_summary(cleaned)
    standardize_check(cleaned)


if __name__ == "__main__":
    main()
