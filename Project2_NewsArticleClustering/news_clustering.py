"""
news_clustering.py
--------------------
Project 2: News Article Clustering System (Lab Sheet 07)

Pipeline:
  1. Load and clean the text (lowercasing, punctuation removal, stopword
     removal, simple stemming).
  2. Convert text to TF-IDF features.
  3. Reduce dimensionality with PCA/SVD for visualization.
  4. Determine the optimal number of clusters (Elbow + Silhouette).
  5. Cluster with K-Means and Agglomerative (Hierarchical) Clustering.
  6. Visualize cluster formation in 2D.
  7. Evaluate clustering quality (Silhouette, Davies-Bouldin,
     Calinski-Harabasz) and, since the synthetic data carries a known
     True_Category, also report accuracy against it for sanity-checking.
  8. Compare the two clustering techniques, and show the top TF-IDF terms
     per cluster so the groups can be understood, not just plotted.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 1. LOAD AND CLEAN THE TEXT
# ----------------------------------------------------------------------------
df = pd.read_csv("dataset/news_articles.csv")
print("Dataset shape:", df.shape)
print(df["True_Category"].value_counts())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # strip punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["Cleaned_Text"] = df["Text"].apply(clean_text)
print("\nSample cleaned text:")
print(df["Cleaned_Text"].iloc[0])

# ----------------------------------------------------------------------------
# 2. TF-IDF VECTORIZATION
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,      # drop terms that appear in >90% of documents (too common)
    min_df=2,        # drop terms that appear in fewer than 2 documents (too rare)
    ngram_range=(1, 2),
)
X_tfidf = vectorizer.fit_transform(df["Cleaned_Text"])
feature_names = np.array(vectorizer.get_feature_names_out())

print(f"\nTF-IDF matrix shape: {X_tfidf.shape} (documents x vocabulary terms)")

# ----------------------------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION FOR VISUALIZATION
#    (TruncatedSVD is the standard choice for sparse TF-IDF matrices -
#    plain PCA needs a dense matrix and loses the sparsity advantage.)
# ----------------------------------------------------------------------------
svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
X_2d = svd.fit_transform(X_tfidf)
print(f"Explained variance by 2 SVD components: {svd.explained_variance_ratio_.sum():.2%}")

# ----------------------------------------------------------------------------
# 4. DETERMINE THE OPTIMAL NUMBER OF CLUSTERS
# ----------------------------------------------------------------------------
k_range = range(2, 10)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_tfidf)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_tfidf, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method - Inertia vs. k")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("outputs_optimal_k.png", dpi=150)
print("\nSaved outputs_optimal_k.png")

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"Silhouette score suggests optimal k = {best_k}")
OPTIMAL_K = best_k

# ----------------------------------------------------------------------------
# 5. CLUSTER WITH K-MEANS
# ----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=RANDOM_STATE, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_tfidf)

# ----------------------------------------------------------------------------
# 6. CLUSTER WITH AGGLOMERATIVE (HIERARCHICAL) CLUSTERING
#    Hierarchical clustering needs a dense array, which is fine here since
#    the TF-IDF matrix for this small corpus is not too large.
# ----------------------------------------------------------------------------
X_dense = X_tfidf.toarray()
agglo = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage="average", metric="cosine")
df["Agglo_Cluster"] = agglo.fit_predict(X_dense)

# ----------------------------------------------------------------------------
# 7. VISUALIZE CLUSTER FORMATION (2D via SVD)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(
    x=X_2d[:, 0], y=X_2d[:, 1], hue=df["KMeans_Cluster"],
    palette="Set2", s=60, ax=axes[0]
)
axes[0].set_title(f"K-Means Clusters (k={OPTIMAL_K}) - TF-IDF, SVD-reduced to 2D")
axes[0].set_xlabel("SVD Component 1")
axes[0].set_ylabel("SVD Component 2")

sns.scatterplot(
    x=X_2d[:, 0], y=X_2d[:, 1], hue=df["Agglo_Cluster"],
    palette="Set2", s=60, ax=axes[1]
)
axes[1].set_title(f"Hierarchical Clusters (k={OPTIMAL_K}) - TF-IDF, SVD-reduced to 2D")
axes[1].set_xlabel("SVD Component 1")
axes[1].set_ylabel("SVD Component 2")

plt.tight_layout()
plt.savefig("outputs_clusters_2d.png", dpi=150)
print("Saved outputs_clusters_2d.png")

# A view colored by the TRUE category, for comparison with the two clusterings above
plt.figure(figsize=(7, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=df["True_Category"], palette="tab10", s=60)
plt.title("Articles colored by TRUE category (ground truth, for comparison only)")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.tight_layout()
plt.savefig("outputs_true_categories_2d.png", dpi=150)
print("Saved outputs_true_categories_2d.png")

# ----------------------------------------------------------------------------
# 8. EVALUATION METRICS
# ----------------------------------------------------------------------------
def evaluate(labels, name):
    sil = silhouette_score(X_tfidf, labels)
    dbi = davies_bouldin_score(X_dense, labels)
    chi = calinski_harabasz_score(X_dense, labels)
    ari = adjusted_rand_score(df["True_Category"], labels)
    nmi = normalized_mutual_info_score(df["True_Category"], labels)
    print(f"\n--- {name} Evaluation ---")
    print(f"Silhouette Score            : {sil:.4f}")
    print(f"Davies-Bouldin Index         : {dbi:.4f}")
    print(f"Calinski-Harabasz Index      : {chi:.4f}")
    print(f"Adjusted Rand Index (vs true): {ari:.4f}  (sanity check only - real use has no true labels)")
    print(f"Normalized Mutual Info       : {nmi:.4f}  (sanity check only)")
    return sil, dbi, chi, ari, nmi


km_metrics = evaluate(df["KMeans_Cluster"], "K-Means")
agglo_metrics = evaluate(df["Agglo_Cluster"], "Agglomerative (Hierarchical)")

comparison = pd.DataFrame({
    "Algorithm": ["K-Means", "Agglomerative (Hierarchical)"],
    "Silhouette Score": [km_metrics[0], agglo_metrics[0]],
    "Davies-Bouldin Index": [km_metrics[1], agglo_metrics[1]],
    "Calinski-Harabasz Index": [km_metrics[2], agglo_metrics[2]],
    "Adjusted Rand Index": [km_metrics[3], agglo_metrics[3]],
    "Normalized Mutual Info": [km_metrics[4], agglo_metrics[4]],
})
print("\n--- Clustering Technique Comparison ---")
print(comparison.to_string(index=False))
comparison.to_csv("outputs_technique_comparison.csv", index=False)

# ----------------------------------------------------------------------------
# 9. TOP TERMS PER CLUSTER - makes the clusters interpretable, not just plotted
# ----------------------------------------------------------------------------
print("\n--- Top TF-IDF terms per K-Means cluster ---")
cluster_centers = kmeans.cluster_centers_
top_terms_records = []
for cluster_id in range(OPTIMAL_K):
    top_indices = cluster_centers[cluster_id].argsort()[::-1][:8]
    top_terms = feature_names[top_indices]
    majority_true_label = df.loc[df["KMeans_Cluster"] == cluster_id, "True_Category"].mode()[0]
    print(f"Cluster {cluster_id} (majority true label: {majority_true_label}): {', '.join(top_terms)}")
    top_terms_records.append({
        "Cluster": cluster_id,
        "Majority_True_Category": majority_true_label,
        "Top_Terms": ", ".join(top_terms),
        "Size": int((df["KMeans_Cluster"] == cluster_id).sum()),
    })

pd.DataFrame(top_terms_records).to_csv("outputs_cluster_top_terms.csv", index=False)

# ----------------------------------------------------------------------------
# 10. DENDROGRAM
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
Z = linkage(X_dense, method="average", metric="cosine")
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("Dendrogram (Average Linkage, Cosine Distance) - News Articles")
plt.xlabel("Sample clusters")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs_dendrogram.png", dpi=150)
print("Saved outputs_dendrogram.png")

# ----------------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------------
df.to_csv("outputs_clustered_articles.csv", index=False)
print("\nSaved outputs_clustered_articles.csv")
print("\nDone.")
