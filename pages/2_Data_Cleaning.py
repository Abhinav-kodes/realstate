import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_loader import load_raw_data
from src.preprocessing import clean_data

st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("Data Cleaning")

df_raw   = load_raw_data()
df_clean, log = clean_data(df_raw)

# ── Cleaning steps ────────────────────────────────────────────────────────────
st.subheader("Cleaning Pipeline")
import pandas as pd
log_df = pd.DataFrame(log)
st.dataframe(log_df, width="stretch", hide_index=True)

c1, c2, c3 = st.columns(3)
c1.metric("Original Rows",  len(df_raw))
c2.metric("Cleaned Rows",   len(df_clean))
c3.metric("Rows Removed",   len(df_raw) - len(df_clean))

# ── Outlier visualisation ─────────────────────────────────────────────────────
st.subheader("Outlier Detection — Price per Area")
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Before Cleaning", "After Cleaning"])
for col_idx, (data, label) in enumerate([(df_raw, "Before"), (df_clean, "After")], 1):
    fig.add_trace(go.Box(
        y=data["price_per_area"], name=label,
        marker_color="#4f98a3" if label == "After" else "#dd6974",
        boxpoints="outliers",
    ), row=1, col=col_idx)
fig.update_layout(template="plotly_dark", height=400, showlegend=False)
st.plotly_chart(fig, width="stretch")

# ── Distribution before vs after ─────────────────────────────────────────────
st.subheader("Price Distribution — Before vs After")
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=df_raw["price_per_area"],   name="Before",
                             marker_color="#dd6974", opacity=0.6, nbinsx=40))
fig2.add_trace(go.Histogram(x=df_clean["price_per_area"], name="After",
                             marker_color="#4f98a3", opacity=0.75, nbinsx=40))
fig2.update_layout(barmode="overlay", template="plotly_dark",
                   xaxis_title="Price per Area", yaxis_title="Count", height=380)
st.plotly_chart(fig2, width="stretch")

# ── IQR explanation ───────────────────────────────────────────────────────────
st.subheader("Null Values Heatmap")
import plotly.figure_factory as ff
null_matrix = df_raw.isnull().astype(int)
if null_matrix.values.sum() == 0:
    st.success("No null values found in the raw dataset.")
else:
    fig3 = px.imshow(null_matrix.T, template="plotly_dark",
                     title="Null Value Map (1 = null)")
    st.plotly_chart(fig3, width="stretch")

with st.expander("Why 1st–99th percentile outlier removal?"):
    st.markdown("""
- Real estate datasets commonly contain **data entry errors** and
  **ultra-premium outliers** that distort regression models.
- Removing the extreme 1% on each tail eliminates these without
  discarding genuine high/low-end properties.
- The `price_per_area` target is the only column trimmed — features are kept intact.
""")