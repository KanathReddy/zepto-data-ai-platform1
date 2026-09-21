from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "analytics" / "titanic.csv"
OUT_DIR = ROOT / "analytics" / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = ROOT / "analytics" / "best_pipeline.joblib"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_model_input(df: pd.DataFrame):
    target = "survived"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_columns(X_train: pd.DataFrame):
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric and c != "survived"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return preprocessor


def evaluate_classifier(name: str, model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([("preprocess", preprocess_columns(X_train)), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm,
    }
    print(f"\n{name} metrics:")
    print(metrics)
    return pipe, metrics


def train_models(X_train, X_test, y_train, y_test):
    results = {}
    for name, estimator in [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=200)),
    ]:
        pipe, metrics = evaluate_classifier(name, estimator, X_train, X_test, y_train, y_test)
        results[name] = (pipe, metrics)
    return results


def visualize_tree(model_name, pipe):
    tree_model = pipe.named_steps["model"]
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(tree_model, feature_names=feature_names, class_names=["No", "Yes"], filled=True, ax=ax)
    fig.savefig(OUT_DIR / f"{model_name.lower().replace(' ', '_')}_tree.png")
    plt.close(fig)


def imbalance_analysis(X_train, X_test, y_train, y_test):
    print("Class balance:", y_train.value_counts(normalize=True).to_dict())
    model = LogisticRegression(max_iter=1000)
    for label, config in [
        ("baseline", {"class_weight": None}),
        ("balanced", {"class_weight": "balanced"}),
    ]:
        estimator = LogisticRegression(max_iter=1000, **config)
        pipe = Pipeline([("preprocess", preprocess_columns(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(label, {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })


def tune_forest(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }
    estimator = RandomForestClassifier(oob_score=True, random_state=42)
    pipe = Pipeline([("preprocess", preprocess_columns(X_train)), ("model", estimator)])
    search = GridSearchCV(pipe, param_grid={"model__" + key: value for key, value in params.items()}, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print("Best RandomForest params:", search.best_params_)
    print("Best OOB score:", best.named_steps["model"].oob_score_)
    return best


def regression_side_task(df: pd.DataFrame):
    features = [c for c in df.columns if c not in ["fare", "survived", "name", "ticket", "cabin", "embarked", "sex", "who", "adult_male", "deck", "embark_town", "alive", "alone"]]
    X = df[features]
    y = df["fare"]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = __import__("sklearn.linear_model").linear_model.LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    residuals = y_test - pred
    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "residual_plot.png")
    plt.close()
    print({"mae": mae, "rmse": rmse, "r2": r2, "adj_r2": adj_r2})


def save_pipeline(results):
    best_name, (best_pipe, _) = max(results.items(), key=lambda kv: kv[1][1]["f1"])
    joblib.dump(best_pipe, MODEL_PATH)
    print("Saved best pipeline to", MODEL_PATH)
    # Reload and sanity check on raw data
    reloaded = joblib.load(MODEL_PATH)
    raw = load_data().drop(columns=["survived"]).iloc[:5]
    pred = reloaded.predict(raw)
    print("Reloaded pipeline prediction shape:", pred.shape)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = build_model_input(df)
    results = train_models(X_train, X_test, y_train, y_test)
    visualize_tree("Decision Tree", results["Decision Tree"][0])
    imbalance_analysis(X_train, X_test, y_train, y_test)
    tune_forest(X_train, X_test, y_train, y_test)
    regression_side_task(df)
    save_pipeline(results)


if __name__ == "__main__":
    main()
