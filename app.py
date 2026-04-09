import streamlit as st
import plotly.graph_objects as go

if "trained" not in st.session_state:
    st.info("First load: training models in background (~20 sec). "
            "All pages will be instant after this.", icon="")
    st.session_state["trained"] = True

st.set_page_config(
    page_title="Real Estate ML",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .step-card {
      background: #1c1b19;
      border: 1px solid #393836;
      border-radius: 10px;
      padding: 14px 10px;
      text-align: center;
      height: 130px;
  }
  .step-icon  { font-size: 26px; }
  .step-title { font-weight: 700; color: #4f98a3; margin: 6px 0 4px; font-size: 13px; }
  .step-desc  { font-size: 11px; color: #797876; line-height: 1.4; }
  .arrow      { display: flex; align-items: center; justify-content: center;
                color: #4f98a3; font-size: 20px; height: 130px; }
</style>
""", unsafe_allow_html=True)

st.title("Real Estate Price Prediction")
st.markdown("#### End-to-end ML pipeline · Geospatial analysis · Explainable AI")
st.markdown("Dataset: **Real Estate Valuation** — 415 properties in Sindian District, New Taipei City, Taiwan")
st.divider()

# ── Pipeline flow ──────────────────────────────────────────────────────────────
st.subheader("ML Pipeline")

steps = [
    ("", "Dataset",             "415 rows · 6 features\nTaipei, Taiwan"),
    ("", "Data Cleaning",       "Outlier removal\nDuplicate check"),
    ("", "Feature Engineering", "20 features\nSpatial · Age · Time"),
    ("", "Geo Clustering",      "K-Means · DBSCAN\n5 spatial zones"),
    ("", "Model Training",      "RF · GBM · XGB · LGBM\n5-fold CV"),
    ("", "Comparison",          "RMSE · MAE · R²\nBest model select"),
    ("", "SHAP XAI",            "Feature impact\nWaterfall · Beeswarm"),
    ("", "Predict",             "Live price estimate\nNew property input"),
]

grid_items = []
for i, (icon, title, desc) in enumerate(steps):
    grid_items.append(("step", icon, title, desc))
    if i < len(steps) - 1:
        grid_items.append(("arrow",))

cols = st.columns(len(grid_items))
for col, item in zip(cols, grid_items):
    with col:
        if item[0] == "step":
            _, icon, title, desc = item
            st.markdown(f"""
            <div class="step-card">
              <div class="step-icon">{icon}</div>
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc.replace(chr(10),'<br>')}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

st.divider()

# ── Quick stats ────────────────────────────────────────────────────────────────
from src.data_loader import load_raw_data
df_raw = load_raw_data()

st.subheader("Dataset at a Glance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Properties",  f"{len(df_raw):,}")
c2.metric("Features",          "6 raw → 21 engineered")
c3.metric("Avg Price",         f"{df_raw['price_per_area'].mean():.1f}")
c4.metric("Price Range",       f"{df_raw['price_per_area'].min():.1f} – {df_raw['price_per_area'].max():.1f}")
c5.metric("MRT Dist Range",    f"{df_raw['dist_mrt'].min():.0f} – {df_raw['dist_mrt'].max():.0f} m")

st.divider()
st.markdown("""
**Navigate using the sidebar** to walk through each stage of the pipeline:

| Page | What you'll see |
|---|---|
| Dataset Overview | Raw data, types, distributions, correlation |
| Data Cleaning | Step-by-step cleaning log, before/after plots |
| Feature Engineering | All 15 engineered features with rationale |
| Geospatial Analysis | Interactive map, K-Means zones, DBSCAN |
| Model Training | Live training, per-model metrics |
| Model Comparison | Side-by-side RMSE / R² / CV bar charts |
| Explainability (SHAP) | Beeswarm, waterfall, dependence plots |
| Predict | Enter a property's details → get a price estimate |
""")