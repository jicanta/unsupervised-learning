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

print("First rows:")
print(df.head())

countries = df["Country"]
X = df.drop(columns=["Country"])
correlation_matrix = X.corr()

features = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

eigenvectors_df = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(features))],
    index=features,
)

print("\nExplained variance by component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print("\nCumulative variance:")
for i, var in enumerate(np.cumsum(explained_variance)):
    print(f"PC1..PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print("\nCorrelation matrix:")
print(correlation_matrix.round(4))

print("\nEigenvalues:")
for i, value in enumerate(eigenvalues):
    print(f"PC{i+1}: {value:.4f}")

print("\nEigenvectors:")
print(eigenvectors_df)


def save_correlation_matrix_plot(matrix):
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Feature Correlation Matrix - Europe")

    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            value = matrix.iat[row, col]
            text_color = "white" if abs(value) > 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")
    fig.tight_layout()

    output_path = IMAGES_DIR / "pca_europe_correlation_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved image: {output_path}")
    plt.close(fig)


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
    print(f"Saved image: {output_path}")
    plt.close(fig)


save_correlation_matrix_plot(correlation_matrix)


save_boxplot(
    X,
    "Original Feature Distribution - Europe",
    "Original value",
    "pca_europe_boxplot_raw.png",
)
save_boxplot(
    X_scaled_df,
    "Standardized Feature Distribution - Europe",
    "Standardized value",
    "pca_europe_boxplot_standardized.png",
)


def save_pc1_index_plot(labels, scores):
    pc1_index = pd.DataFrame({"Country": labels, "PC1": scores[:, 0]}).sort_values("PC1")
    colors = ["crimson" if value < 0 else "seagreen" for value in pc1_index["PC1"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(pc1_index["Country"], pc1_index["PC1"], color=colors, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.9)
    ax.set_title("Countries by PC1 Score")
    ax.set_xlabel("Country")
    ax.set_ylabel("PC1")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output_path = IMAGES_DIR / "pca_europe_pc1_index.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved image: {output_path}")
    plt.close(fig)

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
ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.2f}% variance)")
ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.2f}% variance)")
ax.set_title("PCA - European Countries")
ax.grid(True, alpha=0.3)
fig.tight_layout()

scatter_output = IMAGES_DIR / "pca_europe_scatter.png"
fig.savefig(scatter_output, dpi=300, bbox_inches="tight")
print(f"\nSaved image: {scatter_output}")

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

    ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.2f}% variance)")
    ax.set_title("PCA Biplot - Europe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    biplot_output = IMAGES_DIR / "pca_europe_biplot.png"
    fig.savefig(biplot_output, dpi=300, bbox_inches="tight")
    print(f"Saved image: {biplot_output}")

    plt.show()


biplot(
    X_pca,
    eigenvectors_df.to_numpy(),
    countries,
    features
)

save_pc1_index_plot(countries, X_pca)
