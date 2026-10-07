# Project 2: News Article Clustering System

## What this does
Groups short news articles into topic clusters using **TF-IDF** text
features and **K-Means** / **Agglomerative (Hierarchical) Clustering**, then
compares the two techniques.

## Files
- `generate_dataset.py` - builds `dataset/news_articles.csv`. No real
  dataset was provided, and this environment has no network access to fetch
  a public corpus (e.g. 20 Newsgroups), so this generates 150 short articles
  across 5 realistic topics (Sports, Technology, Politics, Business,
  Health), each with its own vocabulary and sentence templates. A
  `True_Category` column is kept **only** so this assignment's clusters can
  be checked for accuracy - a real system clustering real, unlabeled news
  would not have this column, since that's the whole point of clustering.
- `news_clustering.py` - the full pipeline: clean text -> TF-IDF -> find
  optimal k -> K-Means + Hierarchical clustering -> visualize -> evaluate ->
  compare -> show top terms per cluster.
- `dataset/news_articles.csv` - the dataset used.
- `outputs_*.png` / `outputs_*.csv` - everything the script produces.

## Run
```
python3 generate_dataset.py     # only needed once, to (re)build the dataset
python3 news_clustering.py
```

## Pipeline, step by step
1. **Clean the text**: lowercase, strip punctuation/digits, collapse
   whitespace.
2. **TF-IDF vectorization**: `TfidfVectorizer` with English stopwords
   removed, unigrams + bigrams, and document-frequency filtering
   (`min_df=2`, `max_df=0.9`) to drop overly rare and overly common terms.
3. **Dimensionality reduction for plotting**: `TruncatedSVD` to 2
   components (the standard choice for a sparse TF-IDF matrix - plain PCA
   needs a dense matrix). Note: with this small, templated corpus, 2
   components only capture ~6% of the variance, so the 2D scatter plot
   looks more overlapped than the clustering actually is - all the real
   clustering and evaluation below happens in the full TF-IDF space, not
   the 2D projection.
4. **Optimal number of clusters**: Elbow (inertia) + Silhouette score,
   plotted for k = 2 to 9 (`outputs_optimal_k.png`); the k with the best
   silhouette score is picked automatically.
5. **Cluster with K-Means** and **Agglomerative Clustering** (cosine
   distance, average linkage - the usual choice for text, since cosine
   similarity is what TF-IDF vectors are meant to be compared with).
6. **Visualize**: `outputs_clusters_2d.png` shows both algorithms' clusters
   side by side on the same 2D projection; `outputs_true_categories_2d.png`
   shows the same points colored by their real topic, for comparison.
7. **Evaluate** with Silhouette Score, Davies-Bouldin Index,
   Calinski-Harabasz Index (none need true labels), plus Adjusted Rand
   Index and Normalized Mutual Information against `True_Category` as a
   sanity check specific to this synthetic dataset.
8. **Compare clustering techniques** side by side in
   `outputs_technique_comparison.csv`.
9. **Top terms per cluster** (`outputs_cluster_top_terms.csv`) - the words
   with the highest TF-IDF weight in each cluster's centroid, so you can
   say *what a cluster is about*, not just that it exists. A dendrogram
   (`outputs_dendrogram.png`) shows the hierarchical merge structure.

## Result
Both algorithms recovered the 5 topics exactly (Adjusted Rand Index = 1.0),
which makes sense given how distinct this corpus's vocabulary is per topic -
a real news corpus would be messier and the score would be lower.

## For the viva
- Know what TF-IDF stands for and why it downweights common words (the
  "IDF" part) while still crediting words that repeat a lot in one document
  (the "TF" part).
- Be able to explain why cosine distance/similarity is the natural choice
  for comparing TF-IDF vectors, rather than Euclidean distance.
- Explain why K-Means needs a chosen k upfront while Agglomerative
  Clustering's dendrogram lets you choose it after seeing the merge
  structure.
- Be ready to say, for one cluster, what its top TF-IDF terms are and why
  those make sense for that topic.
- Know why Adjusted Rand Index and NMI are "sanity check only" here - a real
  unsupervised clustering task has no ground-truth labels to compare
  against, which is exactly why Silhouette/Davies-Bouldin/Calinski-Harabasz
  exist (they don't need any).
