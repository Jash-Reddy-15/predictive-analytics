"""
Customer Segmentation Project
==============================
Segments customers based on behavioral and demographic data using
K-Means clustering (scikit-learn). Produces:
  - cleaned dataset with cluster labels
  - cluster profile summary (CSV)
  - visualizations: elbow plot, PCA scatter, radar/heatmap of segment traits
  - a written summary of business insights per segment

Run:
    python segmentation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT = "outputs"

# ----------------------------------------------------------------------
# 1. GENERATE / LOAD DATA
# ----------------------------------------------------------------------
# In a real project, replace this block with:
#   df = pd.read_csv("customers.csv")
# Expected columns: customer_id, age, annual_income, spending_score,
#                    purchase_frequency, recency_days, avg_order_value

np.random.seed(42)
n = 600

def make_segment(n, age, income, spend, freq, recency, aov):
    return pd.DataFrame({
        "age": np.clip(np.random.normal(age, 6, n), 18, 75).astype(int),
        "annual_income": np.clip(np.random.normal(income, 8, n), 15, 200),
        "spending_score": np.clip(np.random.normal(spend, 8, n), 1, 100),
        "purchase_frequency": np.clip(np.random.normal(freq, 2, n), 1, 40).astype(int),
        "recency_days": np.clip(np.random.normal(recency, 8, n), 1, 120).astype(int),
        "avg_order_value": np.clip(np.random.normal(aov, 15, n), 10, 400),
    })

segments_truth = pd.concat([
    make_segment(150, age=28, income=45,  spend=78, freq=18, recency=8,  aov=85),   # Young high spenders
    make_segment(150, age=45, income=110, spend=55, freq=8,  recency=25, aov=220),  # Affluent occasional
    make_segment(150, age=35, income=60,  spend=35, freq=4,  recency=70, aov=60),   # At-risk / low engagement
    make_segment(150, age=52, income=75,  spend=82, freq=22, recency=5,  aov=130),  # Loyal premium
], ignore_index=True)

df = segments_truth.copy()
df.insert(0, "customer_id", [f"CUST{i+1:04d}" for i in range(len(df))])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle

print(f"Loaded {len(df)} customer records with {df.shape[1]-1} features.\n")
print(df.head())

# ----------------------------------------------------------------------
# 2. PREPROCESS
# ----------------------------------------------------------------------
features = ["age", "annual_income", "spending_score",
            "purchase_frequency", "recency_days", "avg_order_value"]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------
# 3. FIND OPTIMAL K (elbow method + silhouette score)
# ----------------------------------------------------------------------
inertias, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(list(K_range), inertias, "o-", color="#378ADD")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster SS)")
plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(list(K_range), sil_scores, "o-", color="#1D9E75")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score by k")

plt.tight_layout()
plt.savefig(f"{OUT}/01_elbow_silhouette.png", dpi=150)
plt.close()

best_k = 4  # chosen based on elbow + silhouette + business interpretability
print(f"\nSelected k = {best_k}")

# ----------------------------------------------------------------------
# 4. FIT FINAL MODEL
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["segment"])
print(f"Silhouette score for k={best_k}: {sil:.3f}")

# ----------------------------------------------------------------------
# 5. PCA FOR 2D VISUALIZATION
# ----------------------------------------------------------------------
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]

palette = {0: "#378ADD", 1: "#1D9E75", 2: "#BA7517", 3: "#D4537E",
           4: "#7A5CC0", 5: "#E0A23C"}

plt.figure(figsize=(7, 6))
for seg in sorted(df["segment"].unique()):
    sub = df[df["segment"] == seg]
    plt.scatter(sub["pc1"], sub["pc2"], s=18, alpha=0.7,
                color=palette.get(seg, "#999"), label=f"Segment {seg}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Customer Segments (PCA projection)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_pca_segments.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. SEGMENT PROFILES
# ----------------------------------------------------------------------
profile = df.groupby("segment")[features].mean().round(1)
profile["customer_count"] = df["segment"].value_counts().sort_index()
profile["pct_of_base"] = (profile["customer_count"] / len(df) * 100).round(1)
print("\nSegment profiles:\n", profile)

profile.to_csv(f"{OUT}/segment_profiles.csv")
df.to_csv(f"{OUT}/customers_segmented.csv", index=False)

# ----------------------------------------------------------------------
# 7. HEATMAP OF NORMALIZED SEGMENT TRAITS
# ----------------------------------------------------------------------
norm_profile = (profile[features] - profile[features].min()) / \
               (profile[features].max() - profile[features].min())

plt.figure(figsize=(8, 4.5))
plt.imshow(norm_profile.values, cmap="YlGnBu", aspect="auto")
plt.colorbar(label="Relative level (0=low, 1=high)")
plt.xticks(range(len(features)), features, rotation=30, ha="right")
plt.yticks(range(best_k), [f"Segment {i}" for i in range(best_k)])
for i in range(best_k):
    for j in range(len(features)):
        plt.text(j, i, f"{profile[features].iloc[i, j]:.0f}",
                 ha="center", va="center", fontsize=9,
                 color="black" if norm_profile.values[i, j] < 0.6 else "white")
plt.title("Segment Characteristics (avg feature values)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_segment_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. AUTO-GENERATE BUSINESS INSIGHT LABELS
# ----------------------------------------------------------------------
def label_segment(row):
    if row["spending_score"] > 65 and row["purchase_frequency"] > 12 and row["age"] < 35:
        return "Young High-Value Spenders"
    if row["annual_income"] > 90 and row["avg_order_value"] > 150:
        return "Affluent Occasional Shoppers"
    if row["recency_days"] > 50:
        return "At-Risk / Churn-Prone"
    if row["purchase_frequency"] > 15 and row["spending_score"] > 65:
        return "Loyal Premium Customers"
    return "General / Mixed"

profile["segment_label"] = profile.apply(label_segment, axis=1)
profile.to_csv(f"{OUT}/segment_profiles.csv")

print("\nSegment labels:")
for seg, row in profile.iterrows():
    print(f"  Segment {seg}: {row['segment_label']} "
          f"({row['customer_count']} customers, {row['pct_of_base']}% of base)")

# ----------------------------------------------------------------------
# 9. WRITE INSIGHTS REPORT
# ----------------------------------------------------------------------
with open(f"{OUT}/insights_report.md", "w") as f:
    f.write("# Customer Segmentation — Insights Report\n\n")
    f.write(f"**Model:** K-Means clustering, k = {best_k}  \n")
    f.write(f"**Silhouette score:** {sil:.3f}  \n")
    f.write(f"**Total customers analyzed:** {len(df)}\n\n")
    f.write("## Segment Summaries\n\n")
    for seg, row in profile.iterrows():
        f.write(f"### Segment {seg}: {row['segment_label']}\n")
        f.write(f"- **Size:** {int(row['customer_count'])} customers "
                f"({row['pct_of_base']}% of base)\n")
        f.write(f"- **Avg age:** {row['age']}\n")
        f.write(f"- **Avg annual income:** ${row['annual_income']}K\n")
        f.write(f"- **Avg spending score:** {row['spending_score']}/100\n")
        f.write(f"- **Avg purchase frequency:** {row['purchase_frequency']} "
                f"orders/period\n")
        f.write(f"- **Avg recency:** {row['recency_days']} days since "
                f"last purchase\n")
        f.write(f"- **Avg order value:** ${row['avg_order_value']}\n\n")

    f.write("## Recommended Actions\n\n")
    f.write("- **Young High-Value Spenders:** Target with trend-driven "
            "campaigns, loyalty points, and referral incentives.\n")
    f.write("- **Affluent Occasional Shoppers:** Focus on premium/exclusive "
            "product lines and personalized high-touch outreach.\n")
    f.write("- **At-Risk / Churn-Prone:** Launch win-back campaigns, "
            "discounts, and re-engagement emails.\n")
    f.write("- **Loyal Premium Customers:** Protect with VIP perks, early "
            "access, and retention rewards — this is your core revenue base.\n")

print(f"\nAll outputs saved to ./{OUT}/")
