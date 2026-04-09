import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import streamlit as st

OPTIMAL_K = 5
ZONE_LABELS = [f"Zone {chr(65+i)}" for i in range(OPTIMAL_K)]


@st.cache_data(show_spinner="Running geospatial clustering …")
def run_clustering(_df: pd.DataFrame):
    coords = _df[["latitude", "longitude"]].values
    scaler_geo = StandardScaler()
    coords_scaled = scaler_geo.fit_transform(coords)

    inertia = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords_scaled).inertia_
        for k in range(2, 11)
    ]

    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(coords_scaled)

    dbscan = DBSCAN(eps=0.015, min_samples=10)
    db_labels = dbscan.fit_predict(coords)
    n_dense = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    return {
        "kmeans": kmeans,
        "scaler_geo": scaler_geo,
        "km_labels": km_labels,
        "db_labels": db_labels,
        "n_dense": n_dense,
        "inertia": inertia,
        "centers_raw": scaler_geo.inverse_transform(kmeans.cluster_centers_),
    }


def add_cluster_cols(_df: pd.DataFrame, cr: dict):
    df = _df.copy()
    df["geo_cluster"]   = cr["km_labels"]
    df["dbscan_cluster"] = cr["db_labels"]
    price_map = df.groupby("geo_cluster")["price_per_area"].mean()
    df["cluster_avg_price"] = df["geo_cluster"].map(price_map)
    return df, price_map