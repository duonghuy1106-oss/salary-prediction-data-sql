# =========================================================
# feature_engineering_modeling.py - BASELINE MODELING
# =========================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/final_salary_prediction_dataset.csv"
TARGET = "annual_median_wage"
OUTPUT_DIR = "outputs/modeling"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def save_plot(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {path}")


def evaluate_model(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred)
    }


# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n========== LOAD DATA ==========")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cannot find dataset at {DATA_PATH}. "
        "Please put final_salary_prediction_dataset.csv inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

print("Dataset shape:", df.shape)


# =========================================================
# 2. FEATURE SELECTION AND BASIC FEATURE ENGINEERING
# =========================================================
print("\n========== FEATURE SELECTION ==========")

# annual_mean_wage is removed to avoid data leakage because it is very close to annual_median_wage.
drop_cols = [
    "soc_code",
    "occupation_title",
    "annual_mean_wage"
]
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols + [TARGET])
y = df[TARGET]

# Keep numeric features only. Non-numeric columns should be encoded earlier in the data preparation step.
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print("Dropping non-numeric columns because they are not encoded:", non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

# Remove columns that are completely missing.
all_missing_cols = X.columns[X.isnull().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns:", all_missing_cols)
    X = X.drop(columns=all_missing_cols)

if X.shape[1] == 0:
    raise ValueError("No usable numeric features found after preprocessing.")

feature_columns = list(X.columns)
feature_means = X.mean(numeric_only=True)

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================
print("\n========== TRAIN TEST SPLIT ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# =========================================================
# 4. DEFINE BASELINE MODELS
# =========================================================
models = {
    "Linear Regression": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=20,
            n_jobs=-1
        ))
    ])
}


# =========================================================
# 5. TRAIN AND EVALUATE
# =========================================================
print("\n========== TRAIN AND EVALUATE ==========")
results = []
predictions = {}

for name, model in models.items():
    print(f"\n----- {name} -----")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_model(y_test, pred)

    results.append({"Model": name, **metrics})
    predictions[name] = pred

    print("MAE:", metrics["MAE"])
    print("RMSE:", metrics["RMSE"])
    print("R2:", metrics["R2"])

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n========== MODEL COMPARISON ==========")
print(results_df)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False, encoding="utf-8-sig")


# =========================================================
# 6. SELECT BEST BASELINE MODEL
# =========================================================
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_pred = predictions[best_model_name]

print("\nBest baseline model:", best_model_name)

joblib.dump(best_model, os.path.join(MODEL_DIR, "baseline_salary_model.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "model_features.pkl"))
joblib.dump(feature_means, os.path.join(MODEL_DIR, "feature_means.pkl"))

print("Saved:")
print("- models/baseline_salary_model.pkl")
print("- models/model_features.pkl")
print("- models/feature_means.pkl")


# =========================================================
# 7. ACTUAL VS PREDICTED
# =========================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"Actual vs Predicted Salary ({best_model_name})")
save_plot("actual_vs_predicted_baseline.png")


# =========================================================
# 8. FEATURE IMPORTANCE FOR RANDOM FOREST
# =========================================================
if "Random Forest" in models:
    rf_model = models["Random Forest"].named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n========== TOP 20 FEATURE IMPORTANCE ==========")
    print(importance_df.head(20))
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False, encoding="utf-8-sig")

    top_features = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importance - Random Forest")
    plt.gca().invert_yaxis()
    save_plot("feature_importance_random_forest.png")


# =========================================================
# 9. SAVE SAMPLE PREDICTIONS
# =========================================================
sample_df = pd.DataFrame({
    "actual_salary": y_test.values,
    "predicted_salary": best_pred,
    "error": y_test.values - best_pred
})
sample_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_prediction_results.csv"), index=False, encoding="utf-8-sig")

print("\n========== BASELINE MODELING COMPLETED ==========")
print(f"Outputs saved to: {OUTPUT_DIR}")
