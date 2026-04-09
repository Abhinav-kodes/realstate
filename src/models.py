import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import streamlit as st

from src.features import FEATURE_COLS, haversine, REF_LAT, REF_LON

# ── Lighter params for cloud deployment ───────────────────────────────────────
MODEL_PARAMS = {
    "Random Forest": dict(
        cls=RandomForestRegressor,
        params=dict(n_estimators=100, max_depth=10, min_samples_split=4,
                    random_state=42, n_jobs=-1),
    ),
    "Gradient Boosting": dict(
        cls=GradientBoostingRegressor,
        params=dict(n_estimators=100, learning_rate=0.1,
                    max_depth=4, subsample=0.8, random_state=42),
    ),
    "XGBoost": dict(
        cls=xgb.XGBRegressor,
        params=dict(n_estimators=100, learning_rate=0.1, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0, n_jobs=-1),
    ),
    "LightGBM": dict(
        cls=lgb.LGBMRegressor,
        params=dict(n_estimators=100, learning_rate=0.1, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1, n_jobs=-1),
    ),
}


# ── Use cache_resource — trains ONCE, persists for entire session ─────────────
@st.cache_resource(show_spinner="Training models — one-time setup, ~20 sec …")
def train_all_models(_df_hash, df_values, df_columns):
    """
    _df_hash   : a hashable key (e.g. len(df)) so cache_resource can key on it
    df_values  : df[FEATURE_COLS + ['price_per_area']].values  (numpy — serialisable)
    df_columns : list of column names
    """
    df = pd.DataFrame(df_values, columns=df_columns)
    X  = df[FEATURE_COLS].copy()
    y  = df["price_per_area"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = {}
    for name, cfg in MODEL_PARAMS.items():
        model = cfg["cls"](**cfg["params"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 3-fold CV instead of 5 — 40% faster
        cv_r2 = cross_val_score(
            cfg["cls"](**cfg["params"]),   # fresh instance for CV
            X_train, y_train, cv=3, scoring="r2", n_jobs=-1
        ).mean()

        results[name] = {
            "model":  model,
            "rmse":   round(float(root_mean_squared_error(y_test, y_pred)), 4),
            "mae":    round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2":     round(float(r2_score(y_test, y_pred)), 4),
            "cv_r2":  round(float(cv_r2), 4),
            "y_pred": y_pred,
        }

    best_name = min(results, key=lambda k: results[k]["rmse"])
    return results, X_train, X_test, y_train, y_test, best_name


@st.cache_resource(show_spinner="Computing SHAP values …")
def get_shap(_model, _X_test_hash, X_test_values, X_test_columns):
    X_test     = pd.DataFrame(X_test_values, columns=X_test_columns)
    explainer  = shap.TreeExplainer(_model)
    shap_vals  = explainer.shap_values(X_test)

    ev = explainer.expected_value
    if hasattr(ev, "__len__"):
        expected_val = float(ev[0]) if len(ev) == 1 else float(np.mean(ev))
    else:
        expected_val = float(ev)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    return shap_vals, expected_val


def predict_single(row_dict, model, kmeans, scaler_geo,
                   cluster_price_map, median_date):
    d = row_dict.copy()
    d["log_dist_mrt"]         = np.log1p(d["dist_mrt"])
    d["sqrt_dist_mrt"]        = np.sqrt(d["dist_mrt"])
    d["mrt_x_stores"]         = d["dist_mrt"] * d["convenience_stores"]
    d["dist_city_center_km"]  = haversine(d["latitude"], d["longitude"],
                                           REF_LAT, REF_LON)
    d["accessibility_score"]  = (d["convenience_stores"] * 10) / \
                                  (np.log1p(d["dist_mrt"]) + 1)
    d["age_squared"]          = d["house_age"] ** 2
    d["is_new"]               = int(d["house_age"] < 5)
    d["is_old"]               = int(d["house_age"] > 20)
    d["age_group"]            = float(min(int(d["house_age"] // 10), 3))
    d["quarter"]              = int(round((d["transaction_date"] % 1) * 4))
    d["is_recent"]            = int(d["transaction_date"] > median_date)
    d["price_zone_lat"]       = 2
    d["price_zone_lon"]       = 2
    c_scaled                  = scaler_geo.transform(
                                    [[d["latitude"], d["longitude"]]])
    d["geo_cluster"]          = int(kmeans.predict(c_scaled)[0])
    d["cluster_avg_price"]    = float(
        cluster_price_map.get(d["geo_cluster"], cluster_price_map.mean()))
    X_new = pd.DataFrame([d])[FEATURE_COLS]
    return float(model.predict(X_new)[0])