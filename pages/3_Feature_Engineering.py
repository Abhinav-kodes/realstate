import streamlit as st
import plotly.express as px
import pandas as pd

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features, ENGINEERED_META, FEATURE_COLS

st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Feature Engineering")
st.markdown("15 new features created across 4 groups: Spatial, Age, Temporal, and Zone.")

df_raw   = load_raw_data()
df_clean, _ = clean_data(df_raw)
df_feat  = engineer_features(df_clean)

# ── Feature catalogue ─────────────────────────────────────────────────────────
st.subheader("Engineered Feature Catalogue")
meta_df = pd.DataFrame(
    [(feat, grp, desc) for feat, (grp, desc) in ENGINEERED_META.items()],
    columns=["Feature", "Group", "Description"],
)
group_filter = st.multiselect("Filter by group:", meta_df["Group"].unique().tolist(),
                               default=meta_df["Group"].unique().tolist())
st.dataframe(meta_df[meta_df["Group"].isin(group_filter)],
             width="stretch", hide_index=True)

# ── Correlation with target ───────────────────────────────────────────────────
st.subheader("Correlation with Target (price_per_area)")
all_feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
corr_target = (
    df_feat[all_feat_cols + ["price_per_area"]]
    .corr()["price_per_area"]
    .drop("price_per_area")
    .sort_values()
)
fig = px.bar(
    x=corr_target.values, y=corr_target.index,
    orientation="h",
    color=corr_target.values,
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    template="plotly_dark",
    title="Pearson correlation of each feature with price_per_area",
    labels={"x": "Correlation", "y": "Feature"},
)
fig.update_layout(height=600, showlegend=False)
st.plotly_chart(fig, width="stretch")

# ── Feature group distributions ───────────────────────────────────────────────
st.subheader("Explore Engineered Features")
feat_choice = st.selectbox("Select feature:", all_feat_cols)
fig2 = px.histogram(df_feat, x=feat_choice, color_discrete_sequence=["#4f98a3"],
                    nbins=40, template="plotly_dark",
                    title=f"Distribution of {feat_choice}")
fig2.update_layout(height=350)
st.plotly_chart(fig2, width="stretch")

# Scatter vs price
fig3 = px.scatter(df_feat, x=feat_choice, y="price_per_area",
                  color="price_per_area", color_continuous_scale="plasma",
                  trendline="ols", trendline_color_override="white",
                  template="plotly_dark",
                  title=f"{feat_choice} vs Price per Area")
fig3.update_layout(height=380)
st.plotly_chart(fig3, width="stretch")