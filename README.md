# The Anatomy of a Livable Neighborhood — Analytics & Modeling

End-to-end **data science workflow** on residential real estate listings (Toronto, Vancouver, Ottawa): custom data collection, cleaning, exploratory analysis, segmentation, **interpretable regression** (hedonic / walkability premium), and **validated ML** with group-aware train/test splits.

---

## Highlights

| Layer | What it does |
|--------|----------------|
| **Data pipeline** | Parses messy listing fields (price, size ranges, beds, tax, lot text), group-wise imputation by `(city, type)`, outlier rules, derived features (`price_per_sqft`, `price_log`, `*_num` aliases). |
| **EDA & market views** | Quality dashboard, distributions, city comparisons, affordability / $-per-sqft snapshots. |
| **Unsupervised** | K-Means **personas** on walk score, $/sqft, and interior size. |
| **Inference** | **OLS** on log-price with property-type and city fixed effects — interpretable **walk score** association after controls. |
| **Prediction** | **Ridge**, **Random Forest**, **Gradient Boosting** on log-price; **`GroupShuffleSplit` by `city`** for holdout evaluation; dollar RMSE (approx.) for reporting. |
| **Interpretation** | Random Forest **feature importances per city** (local market drivers). |

---

## Repository structure

```
├── Housing_Portfolio_Workflow.ipynb   # Main portfolio narrative (run top-to-bottom)
├── listing_pipeline.py                 # Shared prepare_listings() + CSV load helper
├── create_notebook.py                  # Regenerates the portfolio notebook from listing_pipeline.py
├── Group12_MidtermProject - Copy.ipynb # Course / group midterm (scrape + full EDA; large outputs)
└── README.md                           # This file
```

Place your listing **CSV** in this directory (see [Data](#data)).

---

## Requirements

- Python **3.10+** recommended  
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `nbformat` (only if you run `create_notebook.py`)

Install example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nbformat
```

The midterm notebook additionally uses **Selenium**, **BeautifulSoup**, and **webdriver-manager** if you run the scraper cells locally.

---

## Data

The portfolio workflow expects a tabular export with columns compatible with the scrape / clean pipeline, for example:

`address`, `city`, `neighbourhood`, `bed`, `bath`, `size`, `type`, `style`, `walk_score`, `lot_size`, `age`, `listed_price`, `property_tax`, `url`, …

By default, **`Housing_Portfolio_Workflow.ipynb`** looks for the first existing file in:

1. `housing_clean_4b.csv`  
2. `final_data_refactored.csv`  
3. `listings_prepared.csv`

Update the `DATA_CANDIDATES` list in the notebook if your file name differs.

**Typical path:** run the export/save step from the midterm notebook (e.g. `final_data_refactored.csv`), then open the portfolio notebook and **Run All**.

---

## Quick start

1. Clone the repo and add your CSV next to the notebooks.  
2. Open **`Housing_Portfolio_Workflow.ipynb`**.  
3. Execute all cells.  
4. (Optional) After editing **`listing_pipeline.py`**, regenerate the notebook:

```bash
python create_notebook.py
```

---

## Methodology (short)

1. **Prepare** — `prepare_listings()` standardizes types, imputes with group medians where appropriate, applies domain bounds, adds analysis columns.  
2. **Explore** — missingness, univariate and bivariate views (price by city, size vs price vs walk score).  
3. **Segment** — K-Means on scaled walk score, $/sqft, size.  
4. **Infer** — `statsmodels` OLS: `log(price) ~ walk_score + log(size) + baths + C(type) + C(city)`.  
5. **Predict** — scikit-learn pipelines with scaled numerics + one-hot categoricals; **grouped split by `city`**; compare models and report RMSE / MAE / R² (log scale) and approximate dollar RMSE.  
6. **Local drivers** — per-city random forest importances on core numeric features.

---

## Results (what to report)

Metrics are **dataset- and split-dependent**. After a full run, cite:

- OLS **R²** and **walk‑score** effect translated to **~% change per point** (from log specification).  
- Holdout **RMSE / MAE / R²** for each model (Phase 8 table).  
- Cluster summary table and feature-importance comparisons across cities.

**`Housing_Portfolio_Workflow.ipynb`**.

---

