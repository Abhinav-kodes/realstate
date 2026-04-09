import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import streamlit as st

REF_LAT, REF_LON = 25.0330, 121.5654

FEATURE_COLS = [
    "transaction_date", "house_age", "dist_mrt", "convenience_stores",
    "latitude", "longitude",
    "log_dist_mrt", "sqrt_dist_mrt", "mrt_x_stores",
    "dist_city_center_km", "accessibility_score",
    "age_squared", "is_new", "is_old", "age_group",
    "quarter", "is_recent",
    "price_zone_lat", "price_zone_lon",
    "geo_cluster", "cluster_avg_price",
]

ENGINEERED_META = {
    "log_dist_mrt":        ("Spatial",   "log(1 + dist_mrt) – compresses long-tail MRT distances"),
    "sqrt_dist_mrt":       ("Spatial",   "√dist_mrt – alternative non-linear MRT transform"),
    "mrt_x_stores":        ("Spatial",   "dist_mrt × convenience_stores – interaction term"),
    "dist_city_center_km": ("Spatial",   "Haversine distance to Xinyi District, Taipei (km)"),
    "accessibility_score": ("Spatial",   "(stores × 10) / log(1+dist_mrt) – composite walkability"),
    "age_squared":         ("Age",       "house_age² – captures non-linear age depreciation"),
    "is_new":              ("Age",       "1 if house_age < 5 years"),
    "is_old":              ("Age",       "1 if house_age > 20 years"),
    "age_group":           ("Age",       "Quantile bucket (0–3) of house age"),
    "quarter":             ("Temporal",  "Transaction quarter (1–4)"),
    "is_recent":           ("Temporal",  "1 if transaction after dataset median date"),
    "price_zone_lat":      ("Zone",      "Quantile latitude band (0–4)"),
    "price_zone_lon":      ("Zone",      "Quantile longitude band (0–4)"),
    "geo_cluster":         ("Cluster",   "K-Means spatial cluster id (0–4)"),
    "cluster_avg_price":   ("Cluster",   "Mean price of property's spatial cluster"),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi   = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Engineering features …")
def engineer_features(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()

    df["dist_city_center_km"] = haversine(df["latitude"], df["longitude"], REF_LAT, REF_LON)
    df["log_dist_mrt"]        = np.log1p(df["dist_mrt"])
    df["sqrt_dist_mrt"]       = np.sqrt(df["dist_mrt"])
    df["mrt_x_stores"]        = df["dist_mrt"] * df["convenience_stores"]
    df["accessibility_score"] = (df["convenience_stores"] * 10) / (np.log1p(df["dist_mrt"]) + 1)

    df["quarter"]   = ((df["transaction_date"] % 1) * 4).round().astype(int).clip(1, 4)
    df["is_recent"] = (df["transaction_date"] > df["transaction_date"].median()).astype(int)

    df["age_squared"] = df["house_age"] ** 2
    df["is_new"]      = (df["house_age"] < 5).astype(int)
    df["is_old"]      = (df["house_age"] > 20).astype(int)

    df["age_group"]      = pd.qcut(df["house_age"],  q=4, labels=False, duplicates="drop")
    df["price_zone_lat"] = pd.qcut(df["latitude"],   q=5, labels=False, duplicates="drop")
    df["price_zone_lon"] = pd.qcut(df["longitude"],  q=5, labels=False, duplicates="drop")

    base_cols = [c for c in df.columns if c not in ["price_per_area"]]
    imp = SimpleImputer(strategy="median")
    df[base_cols] = imp.fit_transform(df[base_cols])

    return df