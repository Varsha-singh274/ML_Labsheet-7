"""
customer_segmentation.py
--------------------------
Project 1: Customer Segmentation using K-Means Clustering (Lab Sheet 07)

Pipeline:
  1. Load and clean the dataset (missing values, duplicates).
  2. Feature engineering + scaling.
  3. Determine the optimal number of clusters (Elbow method + Silhouette score).
  4. Fit K-Means and assign cluster labels.
  5. Visualize the clusters (2D and pairwise views).
  6. Evaluate with Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
  7. Profile each segment and compare against Agglomerative (Hierarchical)
     Clustering, as the instructions ask to "compare clustering techniques".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 1. LOAD AND CLEAN THE DATASET
# ----------------------------------------------------------------------------
df = pd.read_csv("dataset/Mall_Customers.csv")
print("Raw shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Drop exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Impute missing numeric values with the column median (robust to outliers)
numeric_cols = ["Age", "Annual_Income_k$", "Spending_Score"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nShape after cleaning:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum())

# Encode Gender (not used as a clustering feature here, kept for profiling)
le = LabelEncoder()
df["Gender_Encoded"] = le.fit_transform(df["Gender"])

# ----------------------------------------------------------------------------
# 2. FEATURE SELECTION + SCALING
# ----------------------------------------------------------------------------
features = ["Age", "Annual_Income_k$", "Spending_Score"]
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------------
# 3. DETERMINE THE OPTIMAL NUMBER OF CLUSTERS
#    (a) Elbow method - within-cluster sum of squares (inertia)
#    (b) Silhouette score - how well separated the clusters are
# ----------------------------------------------------------------------------
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method - Inertia vs. k")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("outputs_optimal_k.png", dpi=150)
print("\nSaved outputs_optimal_k.png (elbow curve + silhouette scores)")

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"\nSilhouette score suggests optimal k = {best_k}")
print("(The elbow plot is read visually - the silhouette score picks a concrete k automatically.)")

OPTIMAL_K = best_k  # chosen programmatically from the silhouette score

# ----------------------------------------------------------------------------
# 4. FIT K-MEANS WITH THE CHOSEN K
# ----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nFinal K-Means model fit with k = {OPTIMAL_K}")
print(df["Cluster"].value_counts().sort_index())

# ----------------------------------------------------------------------------
# 5. VISUALIZE THE CLUSTERS
# ----------------------------------------------------------------------------
# (a) Income vs Spending Score - the classic 2D view for this dataset
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="Annual_Income_k$", y="Spending_Score",
    hue="Cluster", palette="Set2", s=70
)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers_original[:, 1], centers_original[:, 2],
    s=250, c="black", marker="X", label="Centroids"
)
plt.title("Customer Segments: Annual Income vs. Spending Score")
plt.legend()
plt.tight_layout()
plt.savefig("outputs_clusters_income_spending.png", dpi=150)
print("Saved outputs_clusters_income_spending.png")

# (b) Pairwise view across all three features
pairplot_fig = sns.pairplot(
    df, vars=features, hue="Cluster", palette="Set2", diag_kind="kde"
)
pairplot_fig.savefig("outputs_clusters_pairplot.png", dpi=150)
print("Saved outputs_clusters_pairplot.png")

# ----------------------------------------------------------------------------
# 6. EVALUATION METRICS
# ----------------------------------------------------------------------------
sil = silhouette_score(X_scaled, df["Cluster"])
dbi = davies_bouldin_score(X_scaled, df["Cluster"])
chi = calinski_harabasz_score(X_scaled, df["Cluster"])

print("\n--- K-Means Evaluation Metrics ---")
print(f"Silhouette Score        : {sil:.4f}  (higher is better, range -1 to 1)")
print(f"Davies-Bouldin Index     : {dbi:.4f}  (lower is better)")
print(f"Calinski-Harabasz Index  : {chi:.4f}  (higher is better)")

# ----------------------------------------------------------------------------
# 7. CLUSTER PROFILING - business insight for each segment
# ----------------------------------------------------------------------------
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
print("\n--- Cluster Profiles (mean values) ---")
print(profile)


def label_segment(row):
    if row["Annual_Income_k$"] >= 70 and row["Spending_Score"] >= 60:
        return "High income, high spending (premium customers)"
    if row["Annual_Income_k$"] >= 70 and row["Spending_Score"] < 40:
        return "High income, low spending (conservative savers)"
    if row["Annual_Income_k$"] < 40 and row["Spending_Score"] >= 60:
        return "Low income, high spending (impulsive spenders)"
    if row["Annual_Income_k$"] < 40 and row["Spending_Score"] < 40:
        return "Low income, low spending (budget-conscious)"
    return "Average income, average spending (standard customers)"


profile["Business Segment"] = profile.apply(label_segment, axis=1)
print("\n--- Business Interpretation ---")
print(profile[["Count", "Business Segment"]])
profile.to_csv("outputs_cluster_profiles.csv")

# ----------------------------------------------------------------------------
# 8. COMPARE CLUSTERING TECHNIQUES: K-Means vs. Agglomerative (Hierarchical)
# ----------------------------------------------------------------------------
agglo = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage="ward")
agglo_labels = agglo.fit_predict(X_scaled)

sil_agglo = silhouette_score(X_scaled, agglo_labels)
dbi_agglo = davies_bouldin_score(X_scaled, agglo_labels)
chi_agglo = calinski_harabasz_score(X_scaled, agglo_labels)

comparison = pd.DataFrame({
    "Algorithm": ["K-Means", "Agglomerative (Hierarchical)"],
    "Silhouette Score": [sil, sil_agglo],
    "Davies-Bouldin Index": [dbi, dbi_agglo],
    "Calinski-Harabasz Index": [chi, chi_agglo],
})
print("\n--- Clustering Technique Comparison ---")
print(comparison.to_string(index=False))
comparison.to_csv("outputs_technique_comparison.csv", index=False)

# Dendrogram for the hierarchical side of the comparison
plt.figure(figsize=(10, 6))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Dendrogram (Ward Linkage) - Hierarchical Clustering")
plt.xlabel("Sample clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs_dendrogram.png", dpi=150)
print("Saved outputs_dendrogram.png")

# ----------------------------------------------------------------------------
# Save the final labeled dataset
# ----------------------------------------------------------------------------
df.to_csv("outputs_segmented_customers.csv", index=False)
print("\nSaved outputs_segmented_customers.csv (original data + assigned Cluster column)")
print("\nDone.")
