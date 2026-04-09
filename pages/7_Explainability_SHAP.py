import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features, FEATURE_COLS
from src.clustering import run_clustering, add_cluster_cols
from src.models import train_all_models, get_shap

st.set_page_config(page_title="Explainability (SHAP)", layout="wide")
st.title("Explainability — SHAP Analysis")
st.markdown("SHAP (SHapley Additive exPlanations) quantifies each feature's contribution "
            "to every individual prediction.")

df_raw  = load_raw_data()
df_clean, _ = clean_data(df_raw)
df_feat = engineer_features(df_clean)
cr      = run_clustering(df_feat)
df, _   = add_cluster_cols(df_feat, cr)
_cols = FEATURE_COLS + ["price_per_area"]
results, X_train, X_test, y_train, y_test, best_name = train_all_models(
    len(df),                        # hashable key
    df[_cols].values,               # numpy array
    _cols,                          # column names
)
# Let user pick model
model_choice = st.selectbox(
    "Select model to explain:",
    list(results.keys()),
    index=list(results.keys()).index(best_name),
)
final_model = results[model_choice]["model"]

with st.spinner("Computing SHAP values …"):
    shap_vals, expected_val = get_shap(
        final_model,
        id(final_model),
        X_test.values,
        list(X_test.columns),
    )

# ── Mean absolute SHAP bar ────────────────────────────────────────────────────
st.subheader("Global Feature Importance (Mean |SHAP|)")
mean_shap = pd.Series(
    np.abs(shap_vals).mean(axis=0),
    index=FEATURE_COLS,
).sort_values(ascending=True)

fig_bar = go.Figure(go.Bar(
    x=mean_shap.values,
    y=mean_shap.index,
    orientation="h",
    marker=dict(
        color=mean_shap.values,
        colorscale="teal",
        showscale=False,
    ),
    text=[f"{v:.4f}" for v in mean_shap.values],
    textposition="outside",
))
fig_bar.update_layout(
    template="plotly_dark", height=600,
    xaxis_title="Mean |SHAP value|",
    title=f"Feature Importance — {model_choice}",
    margin=dict(l=160),
)
st.plotly_chart(fig_bar, width="stretch")

# ── SHAP Beeswarm (matplotlib) ────────────────────────────────────────────────
st.subheader("Beeswarm Summary Plot")
st.markdown("Each dot = one prediction. Position = SHAP value. Colour = feature value.")
fig_bee, ax = plt.subplots(figsize=(10, 7))
fig_bee.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_COLS,
                  plot_type="dot", show=False, max_display=15,
                  color_bar=True)
plt.tight_layout()
st.pyplot(fig_bee, width="stretch")
plt.close()

# ── SHAP Dependence Plots ─────────────────────────────────────────────────────
st.subheader("Dependence Plots — How a Feature Drives SHAP")
top_feats = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=FEATURE_COLS
).nlargest(8).index.tolist()

feat_sel = st.selectbox("Select feature:", top_feats)
fi = FEATURE_COLS.index(feat_sel)

fig_dep = go.Figure()
fig_dep.add_trace(go.Scatter(
    x=X_test[feat_sel].values,
    y=shap_vals[:, fi],
    mode="markers",
    marker=dict(
        color=shap_vals[:, fi],
        colorscale="RdYlGn",
        size=6, opacity=0.7,
        colorbar=dict(title="SHAP value"),
    ),
    text=[f"SHAP={s:.3f}" for s in shap_vals[:, fi]],
))
fig_dep.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig_dep.update_layout(
    template="plotly_dark", height=400,
    xaxis_title=feat_sel.replace("_", " ").title(),
    yaxis_title="SHAP value",
    title=f"SHAP Dependence: {feat_sel}",
)
st.plotly_chart(fig_dep, width="stretch")

# ── Waterfall for individual predictions ─────────────────────────────────────
st.subheader("Waterfall — Single Prediction Explained")
col1, col2 = st.columns(2)
with col1:
    sample_idx = st.slider("Sample index (test set):", 0, len(X_test) - 1, 0)
with col2:
    actual_p    = float(y_test.iloc[sample_idx])
    predicted_p = float(results[model_choice]["y_pred"][sample_idx])
    st.metric("Actual Price",    f"{actual_p:.2f}")
    st.metric("Predicted Price", f"{predicted_p:.2f}",
              delta=f"{predicted_p - actual_p:.2f}")

shap_row = shap_vals[sample_idx]
feat_vals = X_test.iloc[sample_idx]

# Build waterfall manually with Plotly
sorted_idx   = np.argsort(np.abs(shap_row))[-12:]
feat_names_w = [FEATURE_COLS[i] for i in sorted_idx]
shap_w       = [shap_row[i] for i in sorted_idx]

fig_wf = go.Figure(go.Waterfall(
    orientation="h",
    measure=["relative"] * len(shap_w),
    y=feat_names_w,
    x=shap_w,
    connector=dict(line=dict(color="#393836")),
    decreasing=dict(marker=dict(color="#dd6974")),
    increasing=dict(marker=dict(color="#4f98a3")),
    base=expected_val,
))
fig_wf.update_layout(
    template="plotly_dark", height=480,
    xaxis_title="Price contribution",
    title=f"Waterfall — Sample #{sample_idx}  |  Base={expected_val:.2f}",
)
st.plotly_chart(fig_wf, width="stretch")

# ── SHAP Interaction top features ────────────────────────────────────────────
st.subheader("Top Feature SHAP Contributions — Distribution")
top3 = pd.Series(np.abs(shap_vals).mean(axis=0),
                  index=FEATURE_COLS).nlargest(3).index.tolist()
fig_box = go.Figure()
pal = ["#4f98a3", "#dd6974", "#e8af34"]
for i, feat in enumerate(top3):
    fi2 = FEATURE_COLS.index(feat)
    fig_box.add_trace(go.Box(
        y=shap_vals[:, fi2], name=feat,
        marker_color=pal[i], boxpoints="outliers",
    ))
fig_box.update_layout(
    template="plotly_dark", height=400,
    yaxis_title="SHAP value",
    title="SHAP value distribution for top 3 features",
)
st.plotly_chart(fig_box, width="stretch")