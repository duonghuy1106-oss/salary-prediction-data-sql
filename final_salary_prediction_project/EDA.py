# =========================================================
# EDA.py - EXPLORATORY DATA ANALYSIS FOR SALARY PREDICTION
# =========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/final_salary_prediction_dataset.csv"
TARGET = "annual_median_wage"
OUTPUT_DIR = "outputs/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def save_plot(filename: str) -> None:
    """Save current matplotlib figure to outputs/eda."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {path}")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n========== 1. LOAD DATA ==========")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cannot find dataset at {DATA_PATH}. "
        "Please put final_salary_prediction_dataset.csv inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)
require_columns(df, [TARGET])

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 2. BASIC INFORMATION
# =========================================================
print("\n========== 2. BASIC INFORMATION ==========")
print(df.info())

print("\n========== 3. DESCRIPTIVE STATISTICS ==========")
print(df.describe(include="all"))


# =========================================================
# 3. MISSING VALUES AND DUPLICATES
# =========================================================
print("\n========== 4. MISSING VALUES ==========")
missing_df = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": df.isnull().mean() * 100
}).sort_values("missing_count", ascending=False)

print(missing_df.head(30))
missing_df.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"), encoding="utf-8-sig")

print("\n========== 5. DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())
if "soc_code" in df.columns:
    print("Duplicate soc_code:", df["soc_code"].duplicated().sum())


# =========================================================
# 4. TARGET ANALYSIS
# =========================================================
print("\n========== 6. TARGET ANALYSIS ==========")
print(df[TARGET].describe())

plt.figure(figsize=(8, 5))
plt.hist(df[TARGET].dropna(), bins=30)
plt.xlabel("Annual Median Wage")
plt.ylabel("Frequency")
plt.title("Distribution of Annual Median Wage")
save_plot("salary_distribution.png")

plt.figure(figsize=(8, 5))
plt.boxplot(df[TARGET].dropna(), vert=False)
plt.xlabel("Annual Median Wage")
plt.title("Boxplot of Annual Median Wage")
save_plot("salary_boxplot.png")


# =========================================================
# 5. TOP AND LOWEST SALARY OCCUPATIONS
# =========================================================
print("\n========== 7. TOP 10 HIGHEST SALARY OCCUPATIONS ==========")
display_cols = [col for col in ["soc_code", "occupation_title", TARGET] if col in df.columns]

top_salary = df.sort_values(TARGET, ascending=False)[display_cols].head(10)
print(top_salary)
top_salary.to_csv(os.path.join(OUTPUT_DIR, "top_10_highest_salary.csv"), index=False, encoding="utf-8-sig")

if "occupation_title" in df.columns:
    plt.figure(figsize=(10, 6))
    plt.barh(top_salary["occupation_title"], top_salary[TARGET])
    plt.xlabel("Annual Median Wage")
    plt.title("Top 10 Highest Salary Occupations")
    plt.gca().invert_yaxis()
    save_plot("top_10_highest_salary.png")

print("\n========== 8. TOP 10 LOWEST SALARY OCCUPATIONS ==========")
low_salary = df.sort_values(TARGET, ascending=True)[display_cols].head(10)
print(low_salary)
low_salary.to_csv(os.path.join(OUTPUT_DIR, "top_10_lowest_salary.csv"), index=False, encoding="utf-8-sig")

if "occupation_title" in df.columns:
    plt.figure(figsize=(10, 6))
    plt.barh(low_salary["occupation_title"], low_salary[TARGET])
    plt.xlabel("Annual Median Wage")
    plt.title("Top 10 Lowest Salary Occupations")
    plt.gca().invert_yaxis()
    save_plot("top_10_lowest_salary.png")


# =========================================================
# 6. TECHNOLOGY FEATURES ANALYSIS
# =========================================================
print("\n========== 9. TECHNOLOGY FEATURES ==========")
tech_cols = [
    "num_technology_skills",
    "num_technology_categories",
    "num_hot_technology",
    "num_in_demand_technology",
    "has_python",
    "has_sql",
    "has_excel",
    "has_tableau",
    "has_r"
]
tech_cols = [col for col in tech_cols if col in df.columns]

if tech_cols:
    print(df[tech_cols].describe())
    df[tech_cols].describe().to_csv(os.path.join(OUTPUT_DIR, "technology_summary.csv"), encoding="utf-8-sig")
else:
    print("No technology columns found.")

binary_tech_cols = [col for col in ["has_python", "has_sql", "has_excel", "has_tableau", "has_r"] if col in df.columns]

tech_salary_rows = []
for col in binary_tech_cols:
    temp = df.groupby(col)[TARGET].agg(["count", "mean", "median"]).reset_index()
    temp.insert(0, "technology_feature", col)
    tech_salary_rows.append(temp)
    print(f"\nAverage salary by {col}:")
    print(temp)

if tech_salary_rows:
    pd.concat(tech_salary_rows, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "salary_by_technology.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# =========================================================
# 7. CORRELATION ANALYSIS
# =========================================================
print("\n========== 10. CORRELATION WITH SALARY ==========")
numeric_df = df.select_dtypes(include=[np.number])

if TARGET not in numeric_df.columns:
    raise ValueError(f"Target column {TARGET} must be numeric for correlation analysis.")

salary_corr = numeric_df.corr()[TARGET].sort_values(ascending=False)
print("\nTop 20 positive correlations:")
print(salary_corr.head(20))
print("\nTop 20 negative correlations:")
print(salary_corr.tail(20))
salary_corr.to_csv(os.path.join(OUTPUT_DIR, "salary_correlation.csv"), encoding="utf-8-sig")

top_corr = salary_corr.drop(TARGET, errors="ignore").head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_corr.index, top_corr.values)
plt.xlabel("Correlation with Annual Median Wage")
plt.title("Top Features Positively Correlated with Salary")
plt.gca().invert_yaxis()
save_plot("top_positive_correlations.png")

negative_corr = salary_corr.drop(TARGET, errors="ignore").tail(15)
plt.figure(figsize=(10, 6))
plt.barh(negative_corr.index, negative_corr.values)
plt.xlabel("Correlation with Annual Median Wage")
plt.title("Top Features Negatively Correlated with Salary")
plt.gca().invert_yaxis()
save_plot("top_negative_correlations.png")


# =========================================================
# 8. SKILL AND ABILITY ANALYSIS
# =========================================================
print("\n========== 11. SKILL FEATURES ANALYSIS ==========")
skill_cols = [col for col in df.columns if col.startswith("skill_") and col in numeric_df.columns]

if skill_cols:
    skill_corr = df[skill_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nTop 20 skill features related to salary:")
    print(skill_corr.head(20))
    skill_corr.to_csv(os.path.join(OUTPUT_DIR, "skill_correlation.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.barh(skill_corr.head(15).index, skill_corr.head(15).values)
    plt.xlabel("Correlation with Salary")
    plt.title("Top Skill Features Related to Salary")
    plt.gca().invert_yaxis()
    save_plot("top_skill_correlations.png")
else:
    print("No numeric skill columns found.")

print("\n========== 12. ABILITY FEATURES ANALYSIS ==========")
ability_cols = [col for col in df.columns if col.startswith("ability_") and col in numeric_df.columns]

if ability_cols:
    ability_corr = df[ability_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nTop 20 ability features related to salary:")
    print(ability_corr.head(20))
    ability_corr.to_csv(os.path.join(OUTPUT_DIR, "ability_correlation.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.barh(ability_corr.head(15).index, ability_corr.head(15).values)
    plt.xlabel("Correlation with Salary")
    plt.title("Top Ability Features Related to Salary")
    plt.gca().invert_yaxis()
    save_plot("top_ability_correlations.png")
else:
    print("No numeric ability columns found.")


# =========================================================
# 9. TECHNOLOGY VS SALARY SCATTER PLOTS
# =========================================================
print("\n========== 13. TECHNOLOGY VS SALARY ==========")

if "num_technology_skills" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["num_technology_skills"], df[TARGET])
    plt.xlabel("Number of Technology Skills")
    plt.ylabel("Annual Median Wage")
    plt.title("Technology Skills vs Salary")
    save_plot("technology_skills_vs_salary.png")

if "num_hot_technology" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["num_hot_technology"], df[TARGET])
    plt.xlabel("Number of Hot Technologies")
    plt.ylabel("Annual Median Wage")
    plt.title("Hot Technologies vs Salary")
    save_plot("hot_technologies_vs_salary.png")


# =========================================================
# 10. OUTLIER DETECTION USING IQR
# =========================================================
print("\n========== 14. OUTLIER DETECTION ==========")
Q1 = df[TARGET].quantile(0.25)
Q3 = df[TARGET].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[TARGET] < lower_bound) | (df[TARGET] > upper_bound)]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of salary outliers:", outliers.shape[0])

outlier_cols = [col for col in ["soc_code", "occupation_title", TARGET] if col in outliers.columns]
outliers[outlier_cols].sort_values(TARGET, ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "salary_outliers.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 11. SAVE EDA SUMMARY
# =========================================================
print("\n========== 15. SAVE EDA SUMMARY ==========")
eda_summary = pd.DataFrame({
    "metric": [
        "number_of_rows",
        "number_of_columns",
        "average_salary",
        "median_salary",
        "min_salary",
        "max_salary",
        "salary_outliers"
    ],
    "value": [
        df.shape[0],
        df.shape[1],
        df[TARGET].mean(),
        df[TARGET].median(),
        df[TARGET].min(),
        df[TARGET].max(),
        outliers.shape[0]
    ]
})

print(eda_summary)
eda_summary.to_csv(os.path.join(OUTPUT_DIR, "eda_summary.csv"), index=False, encoding="utf-8-sig")

print("\n========== EDA COMPLETED ==========")
print(f"All EDA outputs saved to: {OUTPUT_DIR}")
