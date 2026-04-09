import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data_loader import load_raw_data, FEATURE_META

st.set_page_config(page_title="Dataset Overview", layout="wide")
st.title("Dataset Overview")
st.markdown("Raw dataset before any cleaning or transformation.")

df = load_raw_data()

# ── Basic info ─────────────────────────────────────────────────────────────────
st.subheader("Shape & Types")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows",    len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Nulls",   int(df.isnull().sum().sum()))
c4.metric("Duplicates", int(df.duplicated().sum()))

# Feature dictionary
st.subheader("Feature Dictionary")
meta_df = pd.DataFrame(
    [(col, t, desc) for col, (t, desc) in FEATURE_META.items()],
    columns=["Feature", "Type", "Description"],
)
st.dataframe(meta_df, width="stretch", hide_index=True)

# Raw data sample
st.subheader("Raw Data (first 50 rows)")
st.dataframe(df.head(50), width="stretch")

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.dataframe(df.describe().round(3), width="stretch")

# ── Distributions ──────────────────────────────────────────────────────────────
st.subheader("Feature Distributions")
num_cols = df.columns.tolist()
colors = px.colors.qualitative.Plotly

fig = make_subplots(rows=2, cols=4, subplot_titles=num_cols)
for idx, col in enumerate(num_cols):
    r, c = divmod(idx, 4)
    fig.add_trace(
        go.Histogram(x=df[col], name=col, marker_color=colors[idx],
                     showlegend=False, nbinsx=30),
        row=r + 1, col=c + 1,
    )
fig.update_layout(height=500, template="plotly_dark",
                  margin=dict(t=40, b=20))
st.plotly_chart(fig, width="stretch")

# ── Correlation ────────────────────────────────────────────────────────────────
st.subheader("Correlation Matrix")
corr = df.corr().round(3)
fig2 = px.imshow(
    corr, text_auto=True, color_continuous_scale="RdYlGn",
    zmin=-1, zmax=1, aspect="auto",
    title="Pearson Correlation",
)
fig2.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig2, width="stretch")

# ── Scatter matrix ─────────────────────────────────────────────────────────────
st.subheader("Scatter Matrix (vs Price)")
sel = st.multiselect(
    "Select features to plot against price:",
    [c for c in df.columns if c != "price_per_area"],
    default=["dist_mrt", "convenience_stores", "house_age"],
)
if sel:
    for feat in sel:
        fig3 = px.scatter(
            df, x=feat, y="price_per_area",
            color="price_per_area", color_continuous_scale="plasma",
            trendline="ols", trendline_color_override="#4f98a3",
            template="plotly_dark",
            title=f"Price vs {feat}",
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, width="stretch")