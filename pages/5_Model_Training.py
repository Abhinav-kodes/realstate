import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.features import engineer_features
from src.clustering import run_clustering, add_cluster_cols
from src.models import train_all_models, MODEL_PARAMS

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")

df_raw  = load_raw_data()
df_clean, _ = clean_data(df_raw)
df_feat = engineer_features(df_clean)
cr      = run_clustering(df_feat)
df, _   = add_cluster_cols(df_feat, cr)

results, X_train, X_test, y_train, y_test, best_name = train_all_models(df)

# ── Split info ────────────────────────────────────────────────────────────────
st.subheader("Train / Test Split")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples",  len(X_train) + len(X_test))
c2.metric("Training (80%)", len(X_train))
c3.metric("Testing (20%)",  len(X_test))
c4.metric("Features Used",  X_train.shape[1])

# ── Hyperparameters ───────────────────────────────────────────────────────────
st.subheader("Model Hyperparameters")
for name, cfg in MODEL_PARAMS.items():
    with st.expander(f"⚙️ {name}"):
        st.json(cfg["params"])

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("Training Results")
rows = []
for name, r in results.items():
    rows.append({
        "Model":   name,
        "RMSE ↓":  r["rmse"],
        "MAE ↓":   r["mae"],
        "R² ↑":    r["r2"],
        "CV R² ↑": r["cv_r2"],
        "Best":    "🏆" if name == best_name else "",
    })
results_df = pd.DataFrame(rows)
st.dataframe(results_df, width="stretch", hide_index=True)

st.success(f"**Best Model:** {best_name} — RMSE: {results[best_name]['rmse']} | "
           f"R²: {results[best_name]['r2']}")

# ── Predicted vs actual per model ─────────────────────────────────────────────
st.subheader("Predicted vs Actual — Per Model")
chosen = st.selectbox("Select model:", list(results.keys()), index=list(results.keys()).index(best_name))

r = results[chosen]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_test, y=r["y_pred"], mode="markers",
    marker=dict(color=r["y_pred"], colorscale="viridis", size=6, opacity=0.7),
    name="Predictions",
))
lv = [float(y_test.min()), float(y_test.max())]
fig.add_trace(go.Scatter(x=lv, y=lv, mode="lines",
                          line=dict(color="#dd6974", dash="dash"), name="Perfect fit"))
fig.update_layout(template="plotly_dark", height=430,
                   xaxis_title="Actual Price", yaxis_title="Predicted Price",
                   title=f"{chosen} — Predicted vs Actual  |  R² = {r['r2']}")
st.plotly_chart(fig, width="stretch")

# Residuals
residuals = y_test.values - r["y_pred"]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=r["y_pred"], y=residuals, mode="markers",
    marker=dict(color=residuals, colorscale="RdBu", size=5, opacity=0.65),
))
fig2.add_hline(y=0, line_dash="dash", line_color="#dd6974")
fig2.update_layout(template="plotly_dark", height=380,
                    xaxis_title="Predicted", yaxis_title="Residual",
                    title=f"{chosen} — Residual Plot")
st.plotly_chart(fig2, width="stretch")