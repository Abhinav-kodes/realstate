import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features
from src.clustering import run_clustering, add_cluster_cols, ZONE_LABELS, OPTIMAL_K

st.set_page_config(page_title="Geospatial Analysis", layout="wide")
st.title("Geospatial Analysis")

df_raw    = load_raw_data()
df_clean, _ = clean_data(df_raw)
df_feat   = engineer_features(df_clean)
cr        = run_clustering(df_feat)
df, price_map = add_cluster_cols(df_feat, cr)

# ── Elbow method ──────────────────────────────────────────────────────────────
st.subheader("K-Means Elbow Method")
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=list(range(2, 11)), y=cr["inertia"],
    mode="lines+markers", line=dict(color="#4f98a3", width=2),
    marker=dict(size=8),
))
fig_elbow.add_vline(x=OPTIMAL_K, line_dash="dash", line_color="#dd6974",
                    annotation_text=f"k={OPTIMAL_K} selected")
fig_elbow.update_layout(template="plotly_dark", height=350,
                         xaxis_title="Number of Clusters (k)",
                         yaxis_title="Inertia",
                         title="Elbow Method — Selecting Optimal k")
st.plotly_chart(fig_elbow, width="stretch")

# ── Cluster stats ─────────────────────────────────────────────────────────────
st.subheader("Cluster Profile")
cluster_stats = df.groupby("geo_cluster").agg(
    Zone       =("geo_cluster",     lambda x: ZONE_LABELS[int(x.iloc[0])]),
    Avg_Price  =("price_per_area",  "mean"),
    Median_Price=("price_per_area", "median"),
    Std_Price  =("price_per_area",  "std"),
    Avg_MRT_m  =("dist_mrt",        "mean"),
    Avg_Stores =("convenience_stores","mean"),
    Count      =("price_per_area",  "count"),
).round(2).reset_index(drop=True)
st.dataframe(cluster_stats, width="stretch", hide_index=True)

# ── Plotly cluster scatter ────────────────────────────────────────────────────
st.subheader("Spatial Cluster Map (Plotly)")
df["Zone"] = df["geo_cluster"].apply(lambda x: ZONE_LABELS[int(x)])
fig_sc = px.scatter(
    df, x="longitude", y="latitude",
    color="Zone", size="price_per_area",
    hover_data={"dist_mrt": ":.0f", "convenience_stores": True,
                "price_per_area": ":.2f", "Zone": True},
    template="plotly_dark", height=480,
    title="K-Means Zones — colour=Zone, size=Price",
)
st.plotly_chart(fig_sc, width="stretch")

# ── Folium interactive map ────────────────────────────────────────────────────
st.subheader("Interactive Folium Map")

def price_color(p):
    return "blue" if p < 20 else "green" if p < 35 else "orange" if p < 50 else "red"

m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()],
               zoom_start=13, tiles="CartoDB dark_matter")

HeatMap(
    [[r["latitude"], r["longitude"], r["price_per_area"]] for _, r in df.iterrows()],
    radius=12, blur=15, name="Price Heatmap",
).add_to(m)

mc = MarkerCluster(name="Properties").add_to(m)
for _, r in df.sample(min(250, len(df)), random_state=42).iterrows():
    folium.CircleMarker(
        [r["latitude"], r["longitude"]], radius=5,
        color=price_color(r["price_per_area"]),
        fill=True, fill_opacity=0.75,
        popup=folium.Popup(
            f"<b>Price:</b> {r['price_per_area']:.1f}<br>"
            f"<b>MRT:</b> {r['dist_mrt']:.0f} m<br>"
            f"<b>Stores:</b> {int(r['convenience_stores'])}<br>"
            f"<b>Age:</b> {r['house_age']:.0f} yrs<br>"
            f"<b>{ZONE_LABELS[int(r['geo_cluster'])]}</b>",
            max_width=200),
    ).add_to(mc)

for i, (lat, lon) in enumerate(cr["centers_raw"]):
    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=f'<div style="background:#4f98a3;color:#fff;padding:2px 6px;'
                 f'border-radius:4px;font-size:11px;font-weight:bold;">'
                 f'{ZONE_LABELS[i]}</div>'),
    ).add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width=None, height=500)

# ── DBSCAN ────────────────────────────────────────────────────────────────────
st.subheader("DBSCAN Density Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Dense Regions Found", cr["n_dense"])
c2.metric("Noise Points", int((df["dbscan_cluster"] == -1).sum()))
c3.metric("Clustered Points", int((df["dbscan_cluster"] != -1).sum()))

df["DBSCAN"] = df["dbscan_cluster"].apply(
    lambda x: "Noise" if x == -1 else f"Dense Region {int(x)+1}")
fig_db = px.scatter(
    df, x="longitude", y="latitude", color="DBSCAN",
    template="plotly_dark", height=420,
    title="DBSCAN — Dense regions (eps=0.015, min_samples=10)",
)
st.plotly_chart(fig_db, width="stretch")