import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PCA_DIR = Path(__file__).resolve().parent
DATA_FILE = PCA_DIR.parent / "data" / "europe.csv"
IMAGES_DIR = PCA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

COUNTRY_LABEL_OFFSETS = [
    (6, 6),
    (6, -8),
    (-6, 6),
    (-6, -8),
    (8, 0),
    (0, 8),
    (-8, 0),
    (0, -8),
]

df = pd.read_csv(DATA_FILE)

print("Primeras filas:")
print(df.head())

countries = df["Country"]
X = df.drop(columns=["Country"])

features = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

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


def save_boxplot(data, title, y_label, output_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        [data[column] for column in data.columns],
        tick_labels=data.columns,
        patch_artist=True,
        boxprops={"facecolor": "lightsteelblue", "edgecolor": "steelblue", "alpha": 0.85},
        medianprops={"color": "darkred", "linewidth": 1.5},
        whiskerprops={"color": "slategray"},
        capprops={"color": "slategray"},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "gray",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()

    output_path = IMAGES_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Imagen guardada: {output_path}")
    plt.close(fig)


save_boxplot(
    X,
    "Distribucion de variables originales - Europa",
    "Valor original",
    "pca_europe_boxplot_raw.png",
)
save_boxplot(
    X_scaled_df,
    "Distribucion de variables estandarizadas - Europa",
    "Valor estandarizado",
    "pca_europe_boxplot_standardized.png",
)

def annotate_countries(ax, points, labels):
    for i, label in enumerate(labels):
        dx, dy = COUNTRY_LABEL_OFFSETS[i % len(COUNTRY_LABEL_OFFSETS)]
        ax.annotate(
            label,
            (points[i, 0], points[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="midnightblue",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.2},
        )


fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_pca[:, 0], X_pca[:, 1], color="steelblue", s=35)
annotate_countries(ax, X_pca, countries)

ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.2f}% varianza)")
ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.2f}% varianza)")
ax.set_title("PCA - Países de Europa")
ax.grid(True, alpha=0.3)
fig.tight_layout()

scatter_output = IMAGES_DIR / "pca_europe_scatter.png"
fig.savefig(scatter_output, dpi=300, bbox_inches="tight")
print(f"\nImagen guardada: {scatter_output}")

plt.show()

def biplot(scores, loadings, labels, feature_names):
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(scores[:, 0], scores[:, 1], color="steelblue", s=35)
    annotate_countries(ax, scores, labels)

    scale = 3

    for i, feature in enumerate(feature_names):
        x = loadings[i, 0] * scale
        y = loadings[i, 1] * scale

        ax.arrow(
            0,
            0,
            x,
            y,
            color="crimson",
            alpha=0.85,
            head_width=0.07,
            linewidth=1.2,
            length_includes_head=True,
        )
        ax.text(
            x * 1.1,
            y * 1.1,
            feature,
            fontsize=10,
            color="crimson",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "crimson", "pad": 1.5},
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)

    ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.2f}% varianza)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.2f}% varianza)")
    ax.set_title("Biplot PCA - Europa")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    biplot_output = IMAGES_DIR / "pca_europe_biplot.png"
    fig.savefig(biplot_output, dpi=300, bbox_inches="tight")
    print(f"Imagen guardada: {biplot_output}")

    plt.show()


biplot(
    X_pca,
    pca.components_.T,
    countries,
    features
)
