import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features, FEATURE_COLS
from src.clustering import run_clustering, add_cluster_cols, ZONE_LABELS
from src.models import train_all_models, predict_single

st.set_page_config(page_title="Predict Price", layout="wide")
st.title("Predict House Price")
st.markdown("Enter a property's details to get a real-time price estimate.")

df_raw  = load_raw_data()
df_clean, _ = clean_data(df_raw)
df_feat = engineer_features(df_clean)
cr      = run_clustering(df_feat)
df, price_map = add_cluster_cols(df_feat, cr)
_cols = FEATURE_COLS + ["price_per_area"]
results, X_train, X_test, y_train, y_test, best_name = train_all_models(
    len(df),                        # hashable key
    df[_cols].values,               # numpy array
    _cols,                          # column names
)
best_model = results[best_name]["model"]

median_date = float(df["transaction_date"].median())

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Property Details")
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        transaction_date   = st.number_input(
            "Transaction Date", min_value=2010.0, max_value=2025.0,
            value=2013.5, step=0.25,
            help="e.g. 2013.25 = Q1 2013, 2013.5 = Q2 2013")
        house_age          = st.number_input(
            "House Age (years)", min_value=0.0, max_value=80.0, value=10.0, step=1.0)
    with c2:
        dist_mrt           = st.number_input(
            "Distance to MRT (m)", min_value=0.0, max_value=7000.0, value=400.0, step=50.0)
        convenience_stores = st.number_input(
            "Nearby Convenience Stores", min_value=0, max_value=15, value=5, step=1)
    with c3:
        latitude  = st.number_input("Latitude",  min_value=24.9, max_value=25.1,
                                     value=24.975, step=0.001, format="%.5f")
        longitude = st.number_input("Longitude", min_value=121.4, max_value=121.7,
                                     value=121.540, step=0.001, format="%.5f")

    submitted = st.form_submit_button("Predict Price", width="stretch")

# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    row = {
        "transaction_date":   transaction_date,
        "house_age":          house_age,
        "dist_mrt":           dist_mrt,
        "convenience_stores": convenience_stores,
        "latitude":           latitude,
        "longitude":          longitude,
    }

    with st.spinner("Predicting …"):
        price = predict_single(
            row, best_model,
            cr["kmeans"], cr["scaler_geo"],
            price_map, median_date,
        )

    zone_id   = int(cr["kmeans"].predict(
        cr["scaler_geo"].transform([[latitude, longitude]]))[0])
    zone_label = ZONE_LABELS[zone_id]

    st.divider()
    st.subheader("Prediction Result")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Price",  f"{price:.2f}",  help="10,000 TWD / ping")
    r2.metric("Spatial Zone",     zone_label)
    r3.metric("Zone Avg Price",   f"{price_map[zone_id]:.2f}")
    r4.metric("vs Zone Avg",
              f"{((price - price_map[zone_id]) / price_map[zone_id] * 100):+.1f}%")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        delta={"reference": float(df["price_per_area"].mean()),
               "suffix": " vs dataset avg"},
        title={"text": "Predicted Price (10K TWD/ping)", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, float(df["price_per_area"].max()) * 1.1],
                     "tickcolor": "#797876"},
            "bar":  {"color": "#4f98a3"},
            "steps": [
                {"range": [0, 25],  "color": "#393836"},
                {"range": [25, 45], "color": "#2d2c2a"},
                {"range": [45, 70], "color": "#1c1b19"},
            ],
            "threshold": {
                "line": {"color": "#dd6974", "width": 3},
                "thickness": 0.75,
                "value": float(df["price_per_area"].mean()),
            },
        },
    ))
    fig_gauge.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_gauge, width="stretch")

    # Property profile vs dataset stats
    st.subheader("Your Property vs Dataset")
    compare = pd.DataFrame({
        "Feature": ["House Age", "MRT Distance (m)", "Convenience Stores"],
        "Your Property": [house_age, dist_mrt, convenience_stores],
        "Dataset Mean":  [
            round(df["house_age"].mean(), 1),
            round(df["dist_mrt"].mean(), 1),
            round(df["convenience_stores"].mean(), 1),
        ],
        "Dataset Median": [
            round(df["house_age"].median(), 1),
            round(df["dist_mrt"].median(), 1),
            round(df["convenience_stores"].median(), 1),
        ],
    })
    st.dataframe(compare, width="stretch", hide_index=True)

    # All-model comparison for this property
    st.subheader("All-Model Predictions for This Property")
    preds = {}
    for mname, res in results.items():
        p = predict_single(row, res["model"],
                           cr["kmeans"], cr["scaler_geo"],
                           price_map, median_date)
        preds[mname] = round(p, 2)

    fig_all = go.Figure(go.Bar(
        x=list(preds.keys()), y=list(preds.values()),
        marker_color=["#4f98a3" if k == best_name else "#555555" for k in preds],
        text=[str(v) for v in preds.values()], textposition="outside",
    ))
    fig_all.add_hline(y=float(df["price_per_area"].mean()),
                      line_dash="dash", line_color="#dd6974",
                      annotation_text="Dataset avg")
    fig_all.update_layout(
        template="plotly_dark", height=380,
        yaxis_title="Predicted Price", title="All Model Predictions",
    )
    st.plotly_chart(fig_all, width="stretch")