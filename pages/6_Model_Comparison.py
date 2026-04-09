import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features, FEATURE_COLS
from src.clustering import run_clustering, add_cluster_cols
from src.models import train_all_models

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

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
names  = list(results.keys())
rmses  = [results[n]["rmse"]  for n in names]
maes   = [results[n]["mae"]   for n in names]
r2s    = [results[n]["r2"]    for n in names]
cv_r2s = [results[n]["cv_r2"] for n in names]
colors = ["#4f98a3" if n == best_name else "#555555" for n in names]

# ── Bar charts ────────────────────────────────────────────────────────────────
st.subheader("Metric Comparison")
fig = make_subplots(rows=1, cols=4,
                    subplot_titles=["RMSE ↓", "MAE ↓", "R² ↑", "CV R² ↑"])
for col_idx, vals in enumerate([rmses, maes, r2s, cv_r2s], 1):
    fig.add_trace(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        showlegend=False,
    ), row=1, col=col_idx)
fig.update_layout(template="plotly_dark", height=420, margin=dict(t=50, b=20))
st.plotly_chart(fig, width="stretch")

# ── Radar chart ───────────────────────────────────────────────────────────────
st.subheader("Radar Chart — Normalised Performance")

def normalise(vals, higher_is_better=True):
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.5] * len(vals)
    norm = [(v - mn) / (mx - mn) for v in vals]
    return norm if higher_is_better else [1 - n for n in norm]

cats = ["RMSE (↓)", "MAE (↓)", "R²  (↑)", "CV R² (↑)"]
radar_vals = list(zip(
    normalise(rmses,  higher_is_better=False),
    normalise(maes,   higher_is_better=False),
    normalise(r2s,    higher_is_better=True),
    normalise(cv_r2s, higher_is_better=True),
))

fig2 = go.Figure()
pal  = ["#4f98a3", "#dd6974", "#e8af34", "#6daa45"]
for i, (name, vals) in enumerate(zip(names, radar_vals)):
    v = list(vals) + [vals[0]]
    c = cats + [cats[0]]
    fig2.add_trace(go.Scatterpolar(
        r=v, theta=c, fill="toself", name=name,
        line=dict(color=pal[i], width=2),
        fillcolor=pal[i], opacity=0.2,
    ))
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1],
               tickfont=dict(color="#797876"))),
    template="plotly_dark", height=450,
    title="1.0 = best in class for each metric",
)
st.plotly_chart(fig2, width="stretch")

# ── Residual distribution comparison ─────────────────────────────────────────
st.subheader("Residual Distribution — All Models")
fig3 = go.Figure()
for i, name in enumerate(names):
    residuals = y_test.values - results[name]["y_pred"]
    fig3.add_trace(go.Violin(
        y=residuals, name=name,
        box_visible=True, meanline_visible=True,
        fillcolor=pal[i], line_color=pal[i], opacity=0.7,
    ))
fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
fig3.update_layout(template="plotly_dark", height=420,
                    yaxis_title="Residual (Actual − Predicted)",
                    title="Residual spread per model — tighter = better")
st.plotly_chart(fig3, width="stretch")

# ── Summary table ─────────────────────────────────────────────────────────────
st.subheader("Summary Scorecard")
rows = []
for name in names:
    r = results[name]
    rows.append({
        "Model":   name,
        "RMSE":    r["rmse"],
        "MAE":     r["mae"],
        "R²":      r["r2"],
        "CV R²":   r["cv_r2"],
        "Best":    "🏆" if name == best_name else "",
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.success(f"**Best overall model:** {best_name} — lowest RMSE = {results[best_name]['rmse']}")