# =========================================================
# evaluation_and_tuning.py - MODEL EVALUATION AND TUNING
# =========================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/final_salary_prediction_dataset.csv"
TARGET = "annual_median_wage"
OUTPUT_DIR = "outputs/evaluation"
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


def get_tree_feature_importance(model_pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame | None:
    model = model_pipeline.named_steps.get("model")
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    return None


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
# 2. FEATURE SELECTION
# =========================================================
print("\n========== FEATURE SELECTION ==========")

# Remove ID/title columns and leakage columns.
drop_cols = [
    "soc_code",
    "occupation_title",
    "annual_mean_wage"
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + [TARGET])
y = df[TARGET]

# Use numeric columns only. Encode categorical columns in the data preparation stage if needed.
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print("Dropping non-numeric columns because they are not encoded:", non_numeric_cols)
    X = X.drop(columns=non_numeric_cols)

all_missing_cols = X.columns[X.isnull().all()].tolist()
if all_missing_cols:
    print("Dropping all-missing columns:", all_missing_cols)
    X = X.drop(columns=all_missing_cols)

if X.shape[1] == 0:
    raise ValueError("No usable numeric features found after preprocessing.")

feature_columns = list(X.columns)
feature_means = X.mean(numeric_only=True)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 4. DEFINE MODELS
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
    ]),
    "Gradient Boosting": Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
}


# =========================================================
# 5. TRAIN, CROSS-VALIDATE AND EVALUATE
# =========================================================
print("\n========== TRAIN, CROSS-VALIDATE AND EVALUATE ==========")
results = []
predictions = {}

for name, model in models.items():
    print(f"\n----- {name} -----")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_model(y_test, pred)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    row = {
        "Model": name,
        **metrics,
        "CV Mean R2": cv_scores.mean(),
        "CV Std R2": cv_scores.std()
    }
    results.append(row)
    predictions[name] = pred

    print("MAE:", metrics["MAE"])
    print("RMSE:", metrics["RMSE"])
    print("R2:", metrics["R2"])
    print("CV Mean R2:", cv_scores.mean())

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n========== FINAL RESULTS ==========")
print(results_df)
results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R2 Score")
plt.title("Model Comparison by R2 Score")
save_plot("model_comparison_r2.png")


# =========================================================
# 6. RANDOM FOREST TUNING
# =========================================================
print("\n========== RANDOM FOREST TUNING ==========")

rf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest RF Parameters:")
print(grid_search.best_params_)
print("Best RF CV Score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
best_rf_pred = best_rf.predict(X_test)
best_rf_metrics = evaluate_model(y_test, best_rf_pred)

print("\n========== TUNED RANDOM FOREST ==========")
print("MAE:", best_rf_metrics["MAE"])
print("RMSE:", best_rf_metrics["RMSE"])
print("R2:", best_rf_metrics["R2"])

# Add tuned Random Forest to final comparison.
tuned_row = {
    "Model": "Tuned Random Forest",
    **best_rf_metrics,
    "CV Mean R2": grid_search.best_score_,
    "CV Std R2": np.nan
}
final_results_df = pd.concat([results_df, pd.DataFrame([tuned_row])], ignore_index=True)
final_results_df = final_results_df.sort_values("R2", ascending=False)
final_results_df.to_csv(os.path.join(OUTPUT_DIR, "final_evaluation_results.csv"), index=False, encoding="utf-8-sig")

print("\n========== FINAL RESULTS INCLUDING TUNED MODEL ==========")
print(final_results_df)


# =========================================================
# 7. SELECT AND SAVE BEST MODEL
# =========================================================
best_model_name = final_results_df.iloc[0]["Model"]

if best_model_name == "Tuned Random Forest":
    best_model = best_rf
    best_pred = best_rf_pred
else:
    best_model = models[best_model_name]
    best_pred = predictions[best_model_name]

print("\nBest overall model:", best_model_name)

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_salary_model.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "model_features.pkl"))
joblib.dump(feature_means, os.path.join(MODEL_DIR, "feature_means.pkl"))

print("Saved:")
print("- models/best_salary_model.pkl")
print("- models/model_features.pkl")
print("- models/feature_means.pkl")


# =========================================================
# 8. ACTUAL VS PREDICTED AND RESIDUAL PLOT
# =========================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"Actual vs Predicted Salary ({best_model_name})")
save_plot("actual_vs_predicted_best_model.png")

residuals = y_test - best_pred
plt.figure(figsize=(8, 5))
plt.scatter(best_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.title(f"Residual Plot ({best_model_name})")
save_plot("residual_plot_best_model.png")

prediction_results = pd.DataFrame({
    "actual_salary": y_test.values,
    "predicted_salary": best_pred,
    "error": y_test.values - best_pred,
    "absolute_error": np.abs(y_test.values - best_pred)
})
prediction_results.to_csv(os.path.join(OUTPUT_DIR, "prediction_results.csv"), index=False, encoding="utf-8-sig")


# =========================================================
# 9. FEATURE IMPORTANCE
# =========================================================
importance_df = get_tree_feature_importance(best_model, feature_columns)

if importance_df is not None:
    print("\n========== TOP 20 FEATURES ==========")
    print(importance_df.head(20))
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "best_model_feature_importance.csv"), index=False, encoding="utf-8-sig")

    top_features = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top 15 Feature Importance ({best_model_name})")
    plt.gca().invert_yaxis()
    save_plot("best_model_feature_importance.png")
else:
    print("Best model does not provide tree-based feature importance.")


# =========================================================
# 10. OVERFITTING CHECK
# =========================================================
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
gap = train_score - test_score

overfitting_df = pd.DataFrame({
    "metric": ["train_r2", "test_r2", "gap"],
    "value": [train_score, test_score, gap]
})
overfitting_df.to_csv(os.path.join(OUTPUT_DIR, "overfitting_check.csv"), index=False, encoding="utf-8-sig")

print("\n========== OVERFITTING CHECK ==========")
print("Train R2:", train_score)
print("Test R2:", test_score)
print("Gap:", gap)

if gap > 0.1:
    print("Possible overfitting detected. Consider reducing model complexity or collecting more data.")
else:
    print("Model generalizes reasonably well.")

print("\n========== EVALUATION AND TUNING COMPLETED ==========")
print(f"Outputs saved to: {OUTPUT_DIR}")
