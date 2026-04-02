"""
Listing data preparation — shared by Housing_Portfolio_Workflow.ipynb.
Consolidates parsing, imputation, and outlier rules from the course scraper pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prepare_listings(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean Zolo-style listings into analysis-ready columns (adds *_num aliases)."""
    df = raw.copy()

    if "neighbouhood" in df.columns and "neighbourhood" not in df.columns:
        df = df.rename(columns={"neighbouhood": "neighbourhood"})

    obj = df.select_dtypes(include=["object", "string"]).columns
    if len(obj):
        df[obj] = df[obj].apply(lambda s: s.astype("string").str.strip())
        tokens = {"", "na", "n/a", "none", "no data", "unknown", "xxxxxx"}
        mask = df[obj].apply(lambda s: s.str.strip().str.lower().isin(tokens))
        df[obj] = df[obj].mask(mask, np.nan)

    for c in ["address", "city", "neighbourhood", "type", "style"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.title()

    if "bed" in df.columns:
        m = df["bed"].astype(str).str.extract(r"(\d+)\s*\+\s*(\d+)")
        s = df["bed"].astype(str).str.extract(r"(\d+)")[0]
        df["bed_main"] = pd.to_numeric(m[0], errors="coerce").fillna(pd.to_numeric(s, errors="coerce"))
        df["bed_plus"] = pd.to_numeric(m[1], errors="coerce").fillna(0)
        df["bed_total"] = df["bed_main"].fillna(0) + df["bed_plus"]

    if "size" in df.columns:
        s = (
            df["size"]
            .astype("string")
            .str.lower()
            .str.replace(",", "", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )
        r = s.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
        lo, hi = pd.to_numeric(r[0], errors="coerce"), pd.to_numeric(r[1], errors="coerce")
        lt = pd.to_numeric(s.str.extract(r"<\s*(\d+(?:\.\d+)?)")[0], errors="coerce")
        pl = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)\s*\+")[0], errors="coerce")
        single = pd.to_numeric(s.str.extract(r"^\s*(\d+(?:\.\d+)?)\s*$")[0], errors="coerce")
        df["size_min"] = lo.combine_first(pl)
        df["size_max"] = hi.combine_first(lt)
        df["size_single"] = single
        df["size_mid"] = (
            df[["size_min", "size_max"]]
            .mean(axis=1)
            .fillna(df["size_min"])
            .fillna(df["size_max"])
            .fillna(df["size_single"])
        )

    if "walk_score" in df.columns:
        df["walk_score"] = pd.to_numeric(
            df["walk_score"].replace({"—": np.nan, "–": np.nan}), errors="coerce"
        )
        df.loc[(df["walk_score"] < 0) | (df["walk_score"] > 100), "walk_score"] = np.nan

    if "lot_size" in df.columns:
        s = (
            df["lot_size"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("×", "x")
            .str.replace(r"\s+", " ", regex=True)
        )
        sq = pd.to_numeric(
            s.str.extract(r"(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|sf|ft2|ft²)\b")[0],
            errors="coerce",
        )
        ac = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)\s*a(?:cre)?s?\b")[0], errors="coerce")
        sqm = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)\s*(?:m2|m²|sqm)\b")[0], errors="coerce")
        ha = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)\s*ha\b")[0], errors="coerce")
        dm = s.str.extract(r"(\d+(?:\.\d+)?)\s*[x*]\s*(\d+(?:\.\d+)?)")
        a = pd.to_numeric(dm[0], errors="coerce")
        b = pd.to_numeric(dm[1], errors="coerce")
        dims = (a * b).where(
            ~s.str.contains(r"\b(?:m2|m²|sqm)\b", na=False), (a * b) * 10.7639
        )
        dims = dims.mask((a <= 0) | (b <= 0) | a.isna() | b.isna())
        la = sq.combine_first(ac * 43560).combine_first(sqm * 10.7639).combine_first(ha * 107639).combine_first(dims)
        amb = a.notna() & b.notna() & s.str.contains(r"\b(?:acres?|ha)\b", na=False)
        df["lot_area_sqft"] = la.mask(amb | (la <= 1) | (la > 2_000_000)).astype("Float64")

    if "age" in df.columns:
        age = df["age"].astype("string").str.lower().str.replace(r"\s+", " ", regex=True)
        r = age.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
        lo, hi = pd.to_numeric(r[0], errors="coerce"), pd.to_numeric(r[1], errors="coerce")
        p = pd.to_numeric(age.str.extract(r"(\d+(?:\.\d+)?)\s*\+")[0], errors="coerce")
        sgl = pd.to_numeric(
            age.str.extract(r"(\d+(?:\.\d+)?)(?:\s*(?:years?|yrs?)\b)?$")[0], errors="coerce"
        )
        tmp = lo.combine_first(p).combine_first(sgl).where(hi.isna(), (lo + hi) / 2.0)
        df["age_years"] = tmp.where((tmp >= 0) & (tmp <= 150)).astype("Float64")

    if "listed_price" in df.columns:
        df["listed_price"] = pd.to_numeric(
            df["listed_price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )

    if "property_tax" in df.columns:
        df["property_tax"] = pd.to_numeric(
            df["property_tax"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        )

    if "bath" in df.columns:
        df["bath"] = pd.to_numeric(df["bath"], errors="coerce")

    protected = ["lot_area_sqft", "age_years"]
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in protected]
    if num:
        df[num] = df[num].astype("Float64")
        if {"city", "type"}.issubset(df.columns):
            medg = df.groupby(["city", "type"])[num].transform(lambda s: s.dropna().median())
            df[num] = df[num].fillna(medg)
        df[num] = df[num].fillna(df[num].median(numeric_only=True))

    if "bed_total" in df.columns:
        m = (df["bed_total"] < 0) | (df["bed_total"] > 10)
        df.loc[m, "bed_total"] = np.nan
    if "size_mid" in df.columns:
        m = (df["size_mid"] < 100) | (df["size_mid"] > 20000)
        df.loc[m, "size_mid"] = np.nan
    if "walk_score" in df.columns:
        m = (df["walk_score"] < 0) | (df["walk_score"] > 100)
        df.loc[m, "walk_score"] = np.nan
    if "listed_price" in df.columns:
        m = (df["listed_price"] < 1e4) | (df["listed_price"] > 1e8)
        df.loc[m, "listed_price"] = np.nan
    if {"listed_price", "size_mid"}.issubset(df.columns):
        df["price_per_sqft"] = df["listed_price"] / df["size_mid"]
        m = (df["price_per_sqft"] < 10) | (df["price_per_sqft"] > 5000)
        df.loc[m, "price_per_sqft"] = np.nan
    if {"property_tax", "listed_price"}.issubset(df.columns):
        df.loc[df["property_tax"] == 1, "property_tax"] = np.nan
        m = df["listed_price"].notna() & (df["property_tax"] > 0.10 * df["listed_price"])
        df.loc[m, "property_tax"] = np.nan

    num2 = [c for c in df.select_dtypes(include=[np.number]).columns if c not in protected]
    if num2:
        df[num2] = df[num2].astype("Float64")
        if {"city", "type"}.issubset(df.columns):
            df[num2] = df.groupby(["city", "type"])[num2].transform(lambda s: s.fillna(s.median()))
        df[num2] = df[num2].fillna(df[num2].median(numeric_only=True))

    df["listed_price_num"] = df["listed_price"] if "listed_price" in df.columns else np.nan
    df["walk_score_num"] = df["walk_score"] if "walk_score" in df.columns else np.nan
    df["bath_num"] = df["bath"] if "bath" in df.columns else np.nan
    df["bed_num"] = df["bed_total"] if "bed_total" in df.columns else np.nan
    if "property_tax" in df.columns:
        df["property_tax_num"] = df["property_tax"]

    df["bath_num"] = df["bath_num"].clip(lower=0.5, upper=12)
    df["price_log"] = np.log(df["listed_price_num"].where(df["listed_price_num"] > 0))
    if "lot_area_sqft" in df.columns:
        df["lot_sqft"] = df["lot_area_sqft"]

    return df


def load_prepared_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), low_memory=False)
