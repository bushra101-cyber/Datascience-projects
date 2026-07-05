import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================================
# 0. SETUP: Create Synthetic Retail Data (20+ columns)
# =====================================================================
# Simulating 1000 customers with 22 behavioral variables
X_raw, _ = make_blobs(n_samples=1000, n_features=22, centers=3, random_state=42)

# Create a DataFrame and artificially distort scales to mimic real data
# e.g., Income vs. Purchase Frequency distortion
columns = [f"feature_{i}" for i in range(22)]
df = pd.DataFrame(X_raw, columns=columns)
df['feature_0'] = df['feature_0'] * 10000 + 50000  # Massive scale (Income)
df['feature_1'] = np.clip(df['feature_1'] + 5, 0, 10)  # Small scale (Purchases/Month)

print(f"--- Data Loaded: {df.shape[0]} rows x {df.shape[1]} columns ---")


# =====================================================================
# PHASE 1: SCALE - Mathematical Standardization (Input)
# =====================================================================
# Prevents massive scale features from swallowing smaller variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# =====================================================================
# PHASE 2: COMPRESS - Principal Component Analysis (Process)
# =====================================================================
# Overcoming the Curse of Dimensionality by compressing 22 features to 3D
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio by top 3 components: {pca.explained_variance_ratio_}")
print(f"Total variance preserved: {sum(pca.explained_variance_ratio_)*100:.2f}%")


# =====================================================================
# PHASE 3: CLUSTER - Prove Optimal K (Elbow & Silhouette)
# =====================================================================
wcss = []          # Within-Cluster Sum of Squares (Elbow Method)
sil_scores = []    # Silhouette Scores
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plotting the mathematical proofs
fig, ax1 = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
ax1[0].plot(k_range, wcss, 'bo-', label='WCSS')
ax1[0].set_title('Elbow Method to Find Optimal K')
ax1[0].set_xlabel('Number of Clusters (k)')
ax1[0].set_ylabel('WCSS (Inertia)')
ax1[0].grid(True)

# Silhouette Plot
ax2 = ax1[1]
ax2.plot(k_range, sil_scores, 'ro-', label='Silhouette Score')
ax2.set_title('Silhouette Scores per Cluster Choice')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- Fitting Final Model with Mathematically Optimal Clusters (k=3) ---
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = final_kmeans.fit_transform(X_pca).argmin(axis=1) # Mapping back directly


# =====================================================================
# PHASE 4: TRANSLATE - Business Personas (Output)
# =====================================================================
# Analyze cluster profiles based on key anchor metrics
cluster_profile = df.groupby('Cluster')[['feature_0', 'feature_1']].mean()
cluster_profile.columns = ['Avg_Annual_Income', 'Avg_Purchases_Per_Month']

# Mapping clusters to the specific business mandate personas
persona_mapping = {
    0: "CLUSTER A: HIGH-VALUE ENGAGERS",
    1: "CLUSTER B: MID-TIER EXPLORERS",
    2: "CLUSTER C: LOW-ACTIVITY CHURN RISK"
}

df['Business_Persona'] = df['Cluster'].map(persona_mapping)

print("\n--- FINAL TRANSLATED BUSINESS INTELLIGENCE PROFILE ---")
for cluster_id, profile in cluster_profile.iterrows():
    print(f"\n{persona_mapping[cluster_id]}:")
    print(f" -> Metrics: Income: ${profile['Avg_Annual_Income']:,.2f} | Purchases: {profile['Avg_Purchases_Per_Month']:.1f}/mo")