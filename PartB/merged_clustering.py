import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import f_oneway


# -----------------------------
# 1. Load data
# -----------------------------
df_filtered = pd.read_csv("participant_features.csv")
df_unfiltered = pd.read_csv("../Preprocessed Data/clinical_dataset_preprocessed.csv")


# -----------------------------
# 2. Select columns from unfiltered dataset
#    (fried INCLUDED ONLY for validation)
# -----------------------------
cols_unfiltered = [
    "part_id",
    "gait_speed_4m",
    "gait_get_up",
    "raise_chair_time",
    "activity_regular",
    "age",
    "bmi_score",
    "fried"   # ordinal: 0,1,2 (validation only)
]

df_unfiltered_subset = df_unfiltered[cols_unfiltered]


# -----------------------------
# 3. Merge datasets
# -----------------------------
df = df_filtered.merge(
    df_unfiltered_subset,
    on="part_id",
    how="left"
)

print(f"Merged dataset shape: {df.shape}")


# -----------------------------
# 4. Select clustering features
#    (fried EXCLUDED)
# -----------------------------
features = [
    "gait_speed_4m",
    "gait_get_up",
    "raise_chair_time",
    "activity_regular",
    "age",
    "bmi_score",
    "room_changes_total",
    "room_changes_night",
    # "pct_time_kitchen",
    "pct_time_outdoor"
]



bmi_cutoff = df["bmi_score"].quantile(0.995)
df = df[df["bmi_score"] <= bmi_cutoff].copy()

print(f"Dataset size after BMI outlier removal: {df.shape[0]}")
    
activity_cols = ["room_changes_total", "room_changes_night"]

X = df[features].copy()
X[activity_cols] = np.log1p(X[activity_cols])


# -----------------------------
# 5. Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)


# -----------------------------
# 6. Standardize features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# -----------------------------
# 7. Elbow method
# -----------------------------
inertia = []
k_range = range(2, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for KMeans")
plt.tight_layout()
plt.show()


# -----------------------------
# 8. Fit clustering model
# -----------------------------
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())


# -----------------------------
# 9. Cluster profiling
# -----------------------------
cluster_summary = (
    df.groupby("cluster")[features]
      .mean()
      .round(2)
)

print("\nCluster summary (means):")
print(cluster_summary)


# -----------------------------
# 10. ANOVA on clustering features
# -----------------------------
print("\nANOVA p-values by clustering feature:")
for col in features:
    groups = [
        df[df.cluster == c][col].dropna()
        for c in sorted(df.cluster.unique())
    ]
    pval = f_oneway(*groups).pvalue
    print(f"{col:25s}: {pval:.4e}")


# =====================================================
# FRIED FRAILTY VALIDATION (ORDINAL, NOT USED IN CLUSTERING)
# =====================================================

print("\nMean FRIED score by cluster:")
print(
    df.groupby("cluster")["fried"]
      .mean()
      .round(3)
)

# ANOVA for fried score
fried_groups = [
    df[df.cluster == c]["fried"].dropna()
    for c in sorted(df.cluster.unique())
]

f_stat, pval = f_oneway(*fried_groups)
print(f"\nANOVA p-value for FRIED score: {pval:.4e}")

# -----------------------------
# 11. PCA visualization (side-by-side)
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# ---- Plot 1: Colored by cluster ----
cluster_colors = ["black", "red", "yellow",]

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="cluster",
    palette=cluster_colors,
    ax=axes[0]
)

axes[0].set_title("PCA – Colored by Cluster")
axes[0].legend(title="Cluster")

# ---- Plot 2: Colored by Fried score ----
fried_palette = {
    0: "green",   # robust
    1: "orange",  # pre-frail
    2: "red"      # frail
}

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="fried",
    palette=fried_palette,
    ax=axes[1]
)

axes[1].set_title("PCA – Colored by Fried Frailty")
axes[1].legend(title="Fried")

plt.tight_layout()
plt.show()

# from sklearn.decomposition import PCA
import plotly.express as px

# 3D PCA
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

df["pca_1"] = X_pca3[:, 0]
df["pca_2"] = X_pca3[:, 1]
df["pca_3"] = X_pca3[:, 2]

print("Explained variance ratio:", pca3.explained_variance_ratio_,
      " | Cumulative:", pca3.explained_variance_ratio_.sum())

# ---- Plot 1: Colored by cluster ----
fig_cluster = px.scatter_3d(
    df,
    x="pca_1",
    y="pca_2",
    z="pca_3",
    color="cluster",
    opacity=0.8,
    title="3D PCA – Colored by Cluster"
)
fig_cluster.update_traces(marker=dict(size=5))
fig_cluster.show()

# ---- Plot 2: Colored by FRIED ----
fig_fried = px.scatter_3d(
    df,
    x="pca_1",
    y="pca_2",
    z="pca_3",
    color="fried",
    opacity=0.8,
    color_continuous_scale="Viridis",
    title="3D PCA – Colored by Fried Frailty"
)
fig_fried.update_traces(marker=dict(size=5))
fig_fried.show()


# -----------------------------
# 12. Interpretation plots
# -----------------------------
key_vars = [
    "gait_speed_4m",
    "raise_chair_time",
    "room_changes_total",
    # "pct_time_outdoor"
]

for var in key_vars:
    plt.figure()
    sns.boxplot(data=df, x="cluster", y=var)
    plt.title(f"{var} by cluster")
    plt.tight_layout()
    plt.show()

# Fried frailty visualization
plt.figure()
sns.boxplot(data=df, x="cluster", y="fried")
plt.title("Fried frailty score by cluster")
plt.tight_layout()
plt.show()


# -----------------------------
# 13. Save outputs
# -----------------------------
df.to_csv("clustered_dataset.csv", index=False)
cluster_summary.to_csv("cluster_summary.csv")

print("\nOutputs saved:")
print("- clustered_dataset.csv")
print("- cluster_summary.csv")
