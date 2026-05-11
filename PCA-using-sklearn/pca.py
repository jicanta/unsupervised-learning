import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "europe.csv"

df = pd.read_csv(DATA_FILE)

print("Primeras filas:")
print(df.head())

countries = df["Country"]
X = df.drop(columns=["Country"])

features = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print("\nVarianza explicada por componente:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print("\nVarianza acumulada:")
for i, var in enumerate(np.cumsum(explained_variance)):
    print(f"PC1..PC{i+1}: {var:.4f} ({var*100:.2f}%)")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(features))],
    index=features
)

print("\nCargas de las variables:")
print(loadings)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1])

for i, country in enumerate(countries):
    plt.text(X_pca[i, 0] + 0.05, X_pca[i, 1] + 0.05, country, fontsize=9)

plt.axhline(0, color="gray", linewidth=0.8)
plt.axvline(0, color="gray", linewidth=0.8)
plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% varianza)")
plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% varianza)")
plt.title("PCA - Países de Europa")
plt.grid(True)
plt.tight_layout()
plt.show()

def biplot(scores, loadings, labels, feature_names):
    plt.figure(figsize=(11, 8))

    plt.scatter(scores[:, 0], scores[:, 1])

    for i, label in enumerate(labels):
        plt.text(scores[i, 0] + 0.05, scores[i, 1] + 0.05, label, fontsize=9)

    scale = 3

    for i, feature in enumerate(feature_names):
        x = loadings[i, 0] * scale
        y = loadings[i, 1] * scale

        plt.arrow(0, 0, x, y, head_width=0.07, length_includes_head=True)
        plt.text(x * 1.1, y * 1.1, feature, fontsize=10)

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)

    plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% varianza)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% varianza)")
    plt.title("Biplot PCA - Europa")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


biplot(
    X_pca,
    pca.components_.T,
    countries,
    features
)
