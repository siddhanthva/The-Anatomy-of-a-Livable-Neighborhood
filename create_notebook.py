"""Build Housing_Portfolio_Workflow.ipynb — run: python create_notebook.py"""
from pathlib import Path

import nbformat
import nbformat.v4 as nbf

nb = nbf.new_notebook()

cells = []

def add_md(text: str):
    cells.append(nbf.new_markdown_cell(text.strip()))

def add_code(text: str):
    cells.append(nbf.new_code_cell(text.strip()))

add_md("""
# Canadian Multi-City Housing Listings — Portfolio Workflow

**Signal to hiring managers:** This is a **repeatable analytics workflow**: ingestion → quality checks → EDA → unsupervised segmentation → **interpretable regression** → **validated ML** (group-aware splits).

### What problem does this solve?
We quantify how **walkability**, **dwelling attributes**, and **metro** relate to **asking price**, and we build models whose performance is measured **honestly** (held-out cities in expectation via grouped split).

### Deliverables in this notebook
| Phase | Output |
|-------|--------|
| 1–2 | Config + embedded `prepare_listings()` pipeline |
| 3 | Data-quality dashboard |
| 4 | Core visuals (distribution + relationships) |
| 5 | Market snapshot metrics |
| 6 | K-Means listing personas |
| 7 | OLS on log-price — walk score “premium” |
| 8 | Ridge / Random Forest / Gradient Boosting — **GroupShuffleSplit by `city`** |
| 9 | Feature importance (Random Forest, per city) |
| 10 | Interview-ready recap |
""")

add_md("""
## Phase 1 — Setup

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`

The next cell contains the **full data-prep module** (same source as `listing_pipeline.py` in this folder) so you can submit **one notebook** or keep the `.py` file for import elsewhere.
""")

_BASE = Path(__file__).resolve().parent
_PIPELINE_SRC = (_BASE / "listing_pipeline.py").read_text(encoding="utf-8")

add_code("""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SKStandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")

DATA_CANDIDATES = [
    Path("housing_clean_4b.csv"),
    Path("final_data_refactored.csv"),
    Path("listings_prepared.csv"),
]
""")

add_code(_PIPELINE_SRC.strip())

add_md("""
## Phase 2 — Load and prepare

Set **`DATA_CANDIDATES`** to your export path(s). If you are coming from the midterm notebook, run its **“save CSV”** cell first (e.g. `final_data_refactored.csv`).

`prepare_listings()` harmonizes **raw or partially cleaned** exports into modeling columns (`*_num`, `price_per_sqft`, etc.).
""")

add_code("""
def resolve_dataframe(paths):
    for p in paths:
        if p.exists():
            print(f"Using: {p.resolve()}") 
            raw = pd.read_csv(p, low_memory=False)
            return prepare_listings(raw)
    raise FileNotFoundError(
        "No data file found. Place one of: "
        + ", ".join(str(p) for p in paths)
        + " (or update DATA_CANDIDATES)."
    )

df = resolve_dataframe(DATA_CANDIDATES)
print("Shape:", df.shape)
""")

add_md("## Phase 3 — Data quality dashboard")

add_code("""
qc = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing_pct": (df.isna().mean() * 100).round(2),
})
qc = qc.sort_values("missing_pct", ascending=False)
display(qc.head(20))

key = [
    "listed_price_num", "walk_score_num", "size_mid",
    "bath_num", "bed_num", "city", "type", "price_per_sqft",
]
print("\\nKey columns present:", [c for c in key if c in df.columns])
""")

add_md("""
## Phase 4 — Core EDA

**Principle for portfolios:** a few **high-information** plots beat dozens of repetitive charts.
""")

add_code("""
eda = df.dropna(subset=["listed_price_num", "city"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(eda["listed_price_num"], bins=40, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Listed price (long right tail)")
axes[0].set_xlabel("Price (CAD)")

sns.boxplot(data=eda, x="city", y="listed_price_num", ax=axes[1])
axes[1].set_title("Price by metro")
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()

pair = df.dropna(subset=["size_mid", "listed_price_num", "walk_score_num"])
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(
    pair["size_mid"], pair["listed_price_num"],
    c=pair["walk_score_num"], cmap="viridis", alpha=0.35, s=12
)
plt.colorbar(sc, ax=ax, label="Walk Score")
ax.set_xlabel("Interior size (sq ft, midpoint)")
ax.set_ylabel("Listed price (CAD)")
ax.set_title("Size vs price — coloured by walkability")
plt.tight_layout()
plt.show()
""")

add_md("## Phase 5 — Market snapshot (pandas analytics)")

add_code("""
n = len(df)
snap = pd.Series({
    "listings": n,
    "pct_price_under_1p5m": round(100 * (df["listed_price_num"] < 1_500_000).mean(), 2),
    "pct_walk_ge_90": round(100 * (df["walk_score_num"] >= 90).mean(), 2),
    "mean_price_by_city": df.groupby("city")["listed_price_num"].mean().round(0).to_dict(),
    "cheapest_metro_by_mean_ppsf": (
        df.dropna(subset=["price_per_sqft", "city"])
        .groupby("city")["price_per_sqft"].mean().idxmin()
    ),
})
print(snap.to_string())

top_n = (
    df.dropna(subset=["price_per_sqft", "neighbourhood", "city"])
    .groupby(["city", "neighbourhood"], as_index=False)
    .agg(n=("price_per_sqft", "size"), mean_ppsf=("price_per_sqft", "mean"))
)
top_n = top_n[top_n["n"] >= 10].sort_values("mean_ppsf", ascending=False).head(8)
display(top_n)
""")

add_md("""
## Phase 6 — Unsupervised: listing personas (K-Means)

We segment listings on **walk score**, **price per sqft**, and **size** — sensible axes for urban housing positioning.
""")

add_code("""
cluster_df = df[["walk_score_num", "price_per_sqft", "size_mid"]].dropna()
sample = cluster_df.sample(min(len(cluster_df), 5000), random_state=RANDOM_STATE)
Xs = SKStandardScaler().fit_transform(sample)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
sample = sample.copy()
sample["cluster"] = kmeans.fit_predict(Xs)
summ = (
    sample.groupby("cluster")
    .agg(
        n=("walk_score_num", "count"),
        walk=("walk_score_num", "mean"),
        ppsf=("price_per_sqft", "mean"),
        size=("size_mid", "mean"),
    )
    .round(2)
)
display(summ)
""")

add_md("""
## Phase 7 — OLS: interpretable walkability premium

**Model:** log(price) ~ walk_score + log(size) + baths + property type FE + city FE.

Coefficients are **interpretable** for stakeholders; this complements ML in the next section.
""")

add_code("""
reg_df = df[
    ["listed_price_num", "walk_score_num", "size_mid", "bath_num", "type", "city"]
].dropna()
reg_df = reg_df[(reg_df["listed_price_num"] > 0) & (reg_df["size_mid"] > 0)].copy()
reg_df["log_price"] = np.log(reg_df["listed_price_num"])
reg_df["log_size"] = np.log(reg_df["size_mid"])
reg_df["type"] = reg_df["type"].astype("category")
reg_df["city"] = reg_df["city"].astype("category")

ols = smf.ols(
    "log_price ~ walk_score_num + log_size + bath_num + C(type) + C(city)",
    data=reg_df,
).fit()
print(ols.summary().tables[1])

coef_walk = ols.params.get("walk_score_num", np.nan)
if pd.notna(coef_walk):
    print("\\nInterpretation (~log-linear):")
    print("  Approx % Δ price per +1 walk point:", round((np.exp(coef_walk) - 1) * 100, 3), "%")
    print("  Approx % Δ price per +10 walk points:", round((np.exp(coef_walk * 10) - 1) * 100, 2), "%")
""")

add_md("""
## Phase 8 — Predictive models (portfolio-grade validation)

**Why GroupShuffleSplit by `city`?** A random split can leak **market-level structure** between train and test; grouping encourages **generalization across metros**.

**Target:** `log_price` = `log(listed_price_num)`.

**Metrics:** reported on **held-out group** — RMSE/MAE on log scale and **back-transformed** dollar RMSE for communication.
""")

add_code("""
modeling = df.dropna(
    subset=[
        "listed_price_num", "walk_score_num", "size_mid",
        "bath_num", "bed_num", "type", "city",
    ]
).copy()
modeling = modeling[
    (modeling["listed_price_num"] > 0) & (modeling["size_mid"] > 0)
]
modeling["log_price"] = np.log(modeling["listed_price_num"])

num_cols = ["walk_score_num", "size_mid", "bath_num", "bed_num"]
cat_cols = ["type", "city"]
X = modeling[num_cols + cat_cols]
y = modeling["log_price"]
groups = modeling["city"].astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups))
X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

try:
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", _ohe, cat_cols),
    ]
)


def rmse(y_a, y_b):
    return float(np.sqrt(mean_squared_error(y_a, y_b)))


def evaluate(name, estimator):
    pipe = Pipeline([("prep", preprocess), ("model", estimator)])
    pipe.fit(X_tr, y_tr)
    pred_log = pipe.predict(X_te)
    mae_log = mean_absolute_error(y_te, pred_log)
    r2 = r2_score(y_te, pred_log)
    y_true_usd = np.exp(y_te)
    y_pred_usd = np.clip(np.exp(pred_log), 1, None)
    return {
        "model": name,
        "RMSE_log": rmse(y_te, pred_log),
        "MAE_log": mae_log,
        "R2_log": r2,
        "RMSE_USD_approx": rmse(y_true_usd, y_pred_usd),
    }

rows = [
    evaluate("Ridge", Ridge(alpha=2.0, random_state=RANDOM_STATE)),
    evaluate(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    ),
    evaluate(
        "GradientBoosting",
        GradientBoostingRegressor(random_state=RANDOM_STATE, max_depth=3, n_estimators=200),
    ),
]
metrics_df = pd.DataFrame(rows).round(4)
display(metrics_df)

print("\\nTrain cities:", sorted(modeling.iloc[train_idx]["city"].unique()))
print("Test cities:", sorted(modeling.iloc[test_idx]["city"].unique()))
""")

add_md("## Phase 9 — Feature importance by city (Random Forest)")

add_code("""
rf_results = []
for city_name, grp in df.dropna(subset=["listed_price_num"]).groupby("city"):
    sub = grp[
        ["listed_price_num", "walk_score_num", "size_mid", "bath_num", "bed_num"]
    ].copy().dropna()
    if len(sub) < 200:
        print(f"Skip {city_name}: only {len(sub)} complete rows")
        continue
    Xc = sub[["walk_score_num", "size_mid", "bath_num", "bed_num"]]
    yc = sub["listed_price_num"]
    rf = RandomForestRegressor(
        n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(Xc, yc)
    imps = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values(ascending=False)
    rf_results.append(imps.reset_index().assign(city=city_name))

if rf_results:
    feat_imp = pd.concat(rf_results, ignore_index=True)
    feat_imp.columns = ["feature", "importance", "city"]
    display(feat_imp)
    pivot = feat_imp.pivot(index="feature", columns="city", values="importance").round(4)
    display(pivot)
else:
    print("Not enough per-city rows for Random Forest importance.")
""")

add_md("""
## Phase 10 — Results recap (elevator pitch)

**Analytics**
- Market structure differs by **metro**; **size** and **walk score** move with **ask price** in the expected directions.

**Inference**
- OLS on **log-price** with **type and city fixed effects** gives a **stable, interpretable** walk score association (see Phase 7).

**Prediction**
- **Ridge / RF / Gradient Boosting** are compared with a **group-aware holdout** (Phase 8) so metrics are harder to “game” than a random split.

**Skills demonstrated**
- Pandas data engineering, visualization discipline, `statsmodels` + `sklearn` in one story, and **validation design**.

---

### Optional stretch (not run here)
Spatial cross-validation by neighbourhood, regularized high-cardinality encodings, or survival models if time-on-market appears — natural “Phase 11” talking points in interviews.
""")

nb["cells"] = cells

out = Path(__file__).resolve().parent / "Housing_Portfolio_Workflow.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Wrote", out)
