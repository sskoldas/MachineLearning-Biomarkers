import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform, loguniform
from joblib import dump
from datetime import datetime

from utils import compute_metrics_from_cm, extract_feature_importance

# set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../dataset/early_vs_others.tsv")
results_dir = os.path.join(script_dir, "../results")
models_dir = os.path.join(script_dir, "../models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# load data
df = pd.read_csv(dataset_path, sep="\t")

# extract features
feature_cols = [col for col in df.columns if col not in [
    'BioProject', 'stage', 'mediumBAL', 'mediumNPS', 'mediumSputum',
    'mediumTS', 'adult_pediatric', 'continent', 'sex']]
groups = df['BioProject']
y = df['stage']
meta = df[['mediumBAL', 'mediumNPS', 'mediumSputum', 'mediumTS', 'adult_pediatric', 'continent', 'sex']]

# preprocessing
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df[feature_cols] = df[feature_cols].div(df[feature_cols].sum(axis=1), axis=0)
df[feature_cols] = np.log(df[feature_cols] + 1)

# zero-centering batch correction
for b in df["BioProject"].unique():
        idx = (df["BioProject"] == b)
        batch_features = df.loc[idx, feature_cols]
        batch_means = batch_features.mean(axis=0)
        df.loc[idx, feature_cols] = batch_features - batch_means

X = pd.concat([meta, df[feature_cols]], axis=1)

# define models
logo = LeaveOneGroupOut()
models = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "classifier__n_estimators": randint(100, 1000),
            "classifier__max_depth": randint(5, 20),
            "classifier__min_samples_split": randint(2, 20),
            "classifier__min_samples_leaf": randint(1, 20),
            "classifier__max_features": uniform(0.1, 0.9)
        },
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "classifier__n_neighbors": randint(2, 50),
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        },
    },
    "SVM (RBF)": {
        "model": SVC(kernel='rbf', probability=True),
        "params": {
            "classifier__C": loguniform(1e-2, 1e+2),
            "classifier__gamma": loguniform(1e-2, 1e+2),
            "classifier__shrinking": [True, False]
        },
    },
    "Logistic Regression-L2": {
        "model": LogisticRegression(max_iter=10000),
        "params": {
            "classifier__solver": ["sag"],
            "classifier__C": loguniform(1e-4, 1e+2),
            "classifier__penalty": ["l2"],
            "classifier__fit_intercept": [True],
        },
    },
    "SVM (Linear)": {
        "model": SVC(kernel='linear', probability=True),
        "params": {
            "classifier__C": loguniform(1e-2, 1e+2),
            "classifier__shrinking": [True, False]
        },
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", objective="binary:logistic", n_jobs=4),
        "params": {
            "classifier__n_estimators": randint(100, 1000),
            "classifier__max_depth": randint(5, 15),
            "classifier__learning_rate": loguniform(1e-4, 0.1),
            "classifier__subsample": uniform(0.5, 0.5),
            "classifier__colsample_bytree": uniform(0.5, 0.5),
            "classifier__gamma": loguniform(1e-5, 1e+1),
            "classifier__reg_alpha": loguniform(1e-3, 1e+1),
            "classifier__reg_lambda": loguniform(1e-3, 1e+1)
        }
    },
    "Elastic Net": {
        "model": LogisticRegression(solver="saga", penalty="elasticnet", max_iter=10000),
        "params": {
            "classifier__C": loguniform(1e-1, 2e+1),
            "classifier__l1_ratio": uniform(0, 1),
            "classifier__fit_intercept": [True],
        },
    },
    "Logistic Regression-L1": {
        "model": LogisticRegression(solver="saga", max_iter=10000),
        "params": {
            "classifier__C": loguniform(1e-1, 2e+1),
            "classifier__penalty": ["l1"],
            "classifier__fit_intercept": [True],
        },
    }
}

scoring_metrics = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0)
}

results = {}
labels = [0, 1]
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

for model_name, model_info in models.items():
    print(f"\nTraining: {model_name}")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('smote', SMOTE(k_neighbors=10)),
        ('classifier', model_info["model"])
    ])
    
    random_search = RandomizedSearchCV(
        pipeline,
        model_info["params"],
        n_iter=3, # as example, you may prefer to run it on the server
        cv=logo.split(X, y, groups),
        scoring=scoring_metrics,
        refit="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)

    best_index = random_search.best_index_
    best_params = random_search.best_params_
    best_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('smote', SMOTE(k_neighbors=10)),
        ('classifier', model_info["model"])
    ])
    best_pipeline.set_params(**best_params)

    overall_cm = np.zeros((2, 2))
    fold_metrics = []
    feature_importance_per_fold = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_pipeline.fit(X_train, y_train)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        metrics = compute_metrics_from_cm(cm)
        metrics["fold"] = fold_index
        metrics["test_fold_size"] = len(test_idx)
        fold_metrics.append(metrics)
        overall_cm += cm / len(test_idx)

        importances = extract_feature_importance(best_pipeline, X_test, y_test, X.columns)
        feature_importance_per_fold.append({
             'model': model_name,
             'fold' : fold_index,
             'fold_size': len(test_idx),
             'features': list(importances.keys()),
             'importance_scores': list(importances.values())
        })


    overall_cm /= logo.get_n_splits(groups=groups)
    average_metrics = compute_metrics_from_cm(overall_cm)

    results[model_name] = {
        "best_params": best_params,
        "fold_metrics": fold_metrics,
        "average_metrics": average_metrics,
        "feature_importance_per_fold": feature_importance_per_fold
    }

# Save results
rows = [
    {**m, "model": model}
    for model, data in results.items()
    for m in data["fold_metrics"]
]
df_fold = pd.DataFrame(rows)
df_fold.to_csv(os.path.join(results_dir, f"model_metrics_from_per_fold_{timestamp}.csv"), index=False)

df_avg = pd.DataFrame([
    {**v["average_metrics"], "model": k}
    for k, v in results.items()
])
df_avg.to_csv(os.path.join(results_dir, f"average_model_metrics_{timestamp}.csv"), index=False)

best_params_dict = {model: res["best_params"] for model, res in results.items()}
dump(best_params_dict, os.path.join(models_dir, f"best_params_{timestamp}.joblib"))

importance_rows = []
for model, data in results.items():
    for entry in data.get("feature_importance_per_fold", []):
        for feat, score in zip(entry["features"], entry["importance_scores"]):
            importance_rows.append({
                "model": model,
                "fold": entry["fold"],
                "fold_size": entry["fold_size"],
                "feature": feat,
                "importance": score
            })

df_importances = pd.DataFrame(importance_rows)
df_importances.to_csv(os.path.join(results_dir, f"feature_importance_per_fold_{timestamp}.csv"), index=False)
