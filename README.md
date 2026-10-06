Project 1: Customer Segmentation using K-Means Clustering
What this does
Segments customers into groups based on Age, Annual Income, and Spending Score, using K-Means, and compares the result against Agglomerative (Hierarchical) Clustering as the lab sheet's instructions ask.

Files
generate_dataset.py - builds dataset/Mall_Customers.csv. No real dataset was provided with the lab sheet, so this generates a synthetic one in the style of the well-known "Mall Customers" dataset (same columns), with five genuinely separated customer groups plus a few missing values and one duplicate row so the cleaning step has real work to do.
customer_segmentation.py - the full pipeline: clean -> scale -> find optimal k -> fit K-Means -> visualize -> evaluate -> profile -> compare with Hierarchical Clustering.
dataset/Mall_Customers.csv - the dataset used.
outputs_*.png / outputs_*.csv - everything the script produces (see below).
Run
python3 generate_dataset.py       # only needed once, to (re)build the dataset
python3 customer_segmentation.py
Pipeline, step by step
Clean and preprocess: drop duplicate rows, impute missing numeric values with the column median.
Feature scaling: StandardScaler on Age, Annual Income, Spending Score (K-Means is distance-based, so unscaled income in the hundreds would dominate age/spending on a 0-100 scale).
Optimal number of clusters:
Elbow method - inertia (within-cluster sum of squares) plotted against k from 2 to 10 (outputs_optimal_k.png, left panel).
Silhouette score - plotted the same way (right panel), and the k with the highest score is picked automatically as OPTIMAL_K.
Fit K-Means with OPTIMAL_K clusters (k-means++ init, 10 random restarts).
Visualize:
outputs_clusters_income_spending.png - the classic 2D view (Income vs. Spending Score) with centroids marked.
outputs_clusters_pairplot.png - every pair of features at once.
Evaluate with three metrics that don't need ground-truth labels: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
Profile each cluster's mean Age/Income/Spending and attach a plain-English business label (e.g. "High income, high spending - premium customers") - saved to outputs_cluster_profiles.csv.
Compare clustering techniques: the same scaled data is also run through Agglomerative (Hierarchical) Clustering with Ward linkage, and all three metrics are reported side by side in outputs_technique_comparison.csv, plus a dendrogram (outputs_dendrogram.png).
For the viva
Explain why scaling matters before K-Means (it's distance-based; Annual Income (15-140) would otherwise swamp Spending Score (0-100) and Age (18-65)).
Be able to read the elbow plot yourself, not just quote the silhouette score - the "elbow" is where adding another cluster stops helping much.
Know what Silhouette Score, Davies-Bouldin, and Calinski-Harabasz each measure, and why none of them need the true labels (there are none here - that's what makes this unsupervised).
Be ready to describe each of the 5 segments in one sentence, using the profile table.
