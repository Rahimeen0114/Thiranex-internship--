import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA) PROJECT
# ---------------------------------------------------

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("EXPLORATORY DATA ANALYSIS (EDA) PROJECT")
print("=" * 65)

# 1. LOAD DATASET
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})

print("\n1. DATASET OVERVIEW")
print("-" * 65)
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# 2. DATA QUALITY CHECK
print("\n2. DATA QUALITY CHECK")
print("-" * 65)
missing = df.isnull().sum()
duplicates = df.duplicated().sum()

print("Missing Values:", missing.sum())
print("Duplicate Rows:", duplicates)

summary = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique()
})
summary.to_csv(os.path.join(RESULTS_DIR, "dataset_summary.csv"))

# 3. DESCRIPTIVE STATISTICS
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 65)
stats = df.drop(columns=["diagnosis"]).describe().T
print(stats.head(10))
stats.to_csv(os.path.join(RESULTS_DIR, "descriptive_statistics.csv"))

# 4. CLASS DISTRIBUTION
print("\n4. TARGET CLASS DISTRIBUTION")
print("-" * 65)
print(df["diagnosis"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="diagnosis")
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=300)
plt.close()

# 5. CORRELATION ANALYSIS
print("\n5. CORRELATION ANALYSIS")
print("-" * 65)

numeric_df = df.drop(columns=["diagnosis"])
correlation = numeric_df.corr()

target_corr = correlation["target"].drop("target").abs().sort_values(ascending=False)
print("\nTop 10 Features Most Correlated With Target:")
print(target_corr.head(10))

plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    xticklabels=False,
    yticklabels=False
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# 6. FEATURE DISTRIBUTIONS
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area"
]

plt.figure(figsize=(12, 8))

for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(
        data=df,
        x=feature,
        hue="diagnosis",
        kde=True,
        element="step"
    )
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_distributions.png"), dpi=300)
plt.close()

# 7. PAIRPLOT
pairplot_features = selected_features + ["diagnosis"]
pair_grid = sns.pairplot(
    df[pairplot_features],
    hue="diagnosis",
    corner=True
)
pair_grid.figure.suptitle("Relationship Between Important Features", y=1.02)
pair_grid.savefig(os.path.join(RESULTS_DIR, "pairplot.png"), dpi=200)
plt.close("all")

# 8. STRUCTURED INSIGHTS REPORT
top_features = target_corr.head(5)

report = f"""
EXPLORATORY DATA ANALYSIS REPORT
{'=' * 50}

1. DATASET INFORMATION
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Missing Values: {missing.sum()}
- Duplicate Rows: {duplicates}

2. CLASS DISTRIBUTION
{df["diagnosis"].value_counts().to_string()}

3. KEY INFLUENCING FACTORS
The following features have the strongest correlation with the target:
{top_features.to_string()}

4. MAIN INSIGHTS
- The dataset was checked for missing values and duplicate records.
- Statistical summaries were generated for all numerical features.
- Feature distributions were compared across diagnosis classes.
- Correlation analysis was used to identify influential variables.
- The strongest correlated features can be useful for predictive modeling.

5. CONCLUSION
EDA helps understand the structure, quality, patterns and relationships
within a dataset before building a machine learning model.
"""

with open(os.path.join(RESULTS_DIR, "insights_report.txt"), "w") as file:
    file.write(report)

print("\n6. PROJECT COMPLETED SUCCESSFULLY!")
print("-" * 65)
print("All visualizations and reports are saved in the 'results' folder.")
print(report)
