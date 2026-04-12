# COMP4010 Palmer Penguins Dashboard

This project was developed for `COMP4010 Project 1` using the `Palmer Penguins` dataset. It combines interactive data visualization and basic machine learning to explore penguin morphology, island distribution, and clustering behavior.

The repository includes:
- An interactive dashboard built with `Streamlit`
- Visualization notebooks built with `Plotly`
- A preprocessing notebook
- A machine learning notebook for `K-Means + PCA`
- Pretrained model files so the dashboard can run immediately

## 1. Project Overview

The dashboard is designed to help users:
- Explore penguin records by `Species`, `Island`, and `Sex`
- Compare the distributions of bill length, bill depth, flipper length, and body mass
- Analyze correlations between biological measurements
- Illustrate `Simpson's Paradox` in a biological dataset
- Apply `K-Means` clustering and `PCA` dimensionality reduction
- Compare predicted cluster labels with true species labels in `Compare View`

## 2. Repository Structure

```text
comp4010-penguins-dashboard/
├── penguins_dashboard.py              # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── README.md
├── data/
│   ├── load_penguins_data.py          # Script to fetch Palmer Penguins data
│   ├── penguins.csv                   # Source dataset
│   ├── penguins_raw.csv               # Raw source dataset
│   └── processed/
│       ├── penguins_cleaned.csv       # Cleaned dataset used by the dashboard
│       └── penguins_raw_cleaned.csv
└── ML_implementation/
    ├── ML.ipynb                       # Machine learning training notebook
    ├── scaler.pkl                     # Saved StandardScaler
    ├── pca.pkl                        # Saved PCA model
    ├── kmeans_model.pkl               # Saved KMeans model
    └── clustering_results.csv         # Exported clustering results
└── notebook/
    ├── plotly_viz.ipynb                   # Plotly visualization notebook
    ├── preprocessing.ipynb                # Data preprocessing notebook

```
## 3. Reproducing The Full Pipeline

If you want to rerun the full workflow from scratch:

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Reload the source data if needed:

```powershell
python .\data\load_penguins_data.py
```

3. Run the preprocessing notebook:

- `notebook/preprocessing.ipynb`

4. Run the machine learning notebook:

- `ML_implementation/ML.ipynb`

5. Launch the dashboard:

```powershell
streamlit run penguins_dashboard.py
```

## 4. Dashboard Tabs

The dashboard contains 4 main tabs.

### `Overview`
- High-level dataset summary
- Species distribution by island
- Species proportions
- General descriptive insights

### `Distributions`
- Histograms
- Box plots
- Distribution comparisons for key morphological variables

### `Correlations`
- Scatter plots
- Correlation heatmap
- Simpson's Paradox example
- Multi-axis explorer for custom variable comparisons

### `ML Explorer`
- Applies the pretrained `K-Means` model
- Projects the filtered data into `PC1` and `PC2` using `PCA`
- Compares `Cluster Labels`, `Ground Truth`, and `Compare View`
- Reports `Silhouette Score`, `Adjusted Rand Index`, and mismatch count
