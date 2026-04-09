import os, joblib, hashlib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from src.features import FEATURE_COLS, haversine, REF_LAT, REF_LON

CACHE_PATH = "/tmp/re_model_cache.pkl"

MODEL_PARAMS = {
    "Random Forest": dict(
        cls=RandomForestRegressor,
        params=dict(n_estimators=50, max_depth=8,
                    random_state=42, n_jobs=1),
    ),
    "XGBoost": dict(
        cls=xgb.XGBRegressor,
        params=dict(n_estimators=50, learning_rate=0.1, max_depth=5,
                    subsample=0.8, random_state=42,
                    verbosity=0, n_jobs=1),
    ),
    "LightGBM": dict(
        cls=lgb.LGBMRegressor,
        params=dict(n_estimators=50, learning_rate=0.1, num_leaves=31,
                    subsample=0.8, random_state=42,
                    verbose=-1, n_jobs=1),
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


# ── Training ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Training models — first load only (~20 sec)…")
def train_all_models(_hash_key, df_values, df_columns):
    # Try disk cache first
    if os.path.exists(CACHE_PATH):
        try:
            cached = joblib.load(CACHE_PATH)
            if cached.get("hash") == _hash_key:
                return (cached["results"], cached["X_train"], cached["X_test"],
                        cached["y_train"], cached["y_test"], cached["best_name"])
        except Exception:
            pass  # corrupt cache → retrain

    df = pd.DataFrame(df_values, columns=df_columns)
    X  = df[FEATURE_COLS]
    y  = df["price_per_area"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = {}
    for name, cfg in MODEL_PARAMS.items():
        model = cfg["cls"](**cfg["params"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        results[name] = {
            "model":  model,
            "rmse":   round(rmse, 4),
            "mae":    round(float(mean_absolute_error(y_test, y_pred)), 4),
            "r2":     round(float(r2_score(y_test, y_pred)), 4),
            "cv_r2":  None,   # skipped on cloud for speed
            "y_pred": y_pred,
        }

    best_name = min(results, key=lambda k: results[k]["rmse"])

    joblib.dump({
        "hash": _hash_key, "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "best_name": best_name,
    }, CACHE_PATH)

    return results, X_train, X_test, y_train, y_test, best_name


# ── SHAP ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Computing SHAP values…")
def get_shap(_model, _hash_key, X_test_values, X_test_columns):
    X_test    = pd.DataFrame(X_test_values, columns=X_test_columns)
    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer.shap_values(X_test)

    # Normalise expected_value — handles scalar, list, and array
    ev = explainer.expected_value
    if hasattr(ev, "__len__"):
        expected_val = float(ev[0]) if len(ev) == 1 else float(np.mean(ev))
    else:
        expected_val = float(ev)

    # Some versions return list-of-arrays for regressors
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    return shap_vals, expected_val


# ── Single prediction ─────────────────────────────────────────────────────────
def predict_single(row_dict, model, kmeans, scaler_geo,
                   cluster_price_map, median_date):
    d = row_dict.copy()

    # ── Normalise price_map to plain dict once — handles Series, ndarray, dict
    if isinstance(cluster_price_map, dict):
        price_dict = cluster_price_map
    elif isinstance(cluster_price_map, pd.Series):
        price_dict = cluster_price_map.to_dict()          # {0: 35.2, 1: 42.1 …}
    elif isinstance(cluster_price_map, np.ndarray):
        price_dict = {i: float(v) for i, v in enumerate(cluster_price_map)}
    else:
        price_dict = dict(cluster_price_map)              # fallback

    d["log_dist_mrt"]        = np.log1p(d["dist_mrt"])
    d["sqrt_dist_mrt"]       = np.sqrt(d["dist_mrt"])
    d["mrt_x_stores"]        = d["dist_mrt"] * d["convenience_stores"]
    d["dist_city_center_km"] = haversine(d["latitude"], d["longitude"],
                                          REF_LAT, REF_LON)
    d["accessibility_score"] = (d["convenience_stores"] * 10) / \
                                (np.log1p(d["dist_mrt"]) + 1)
    d["age_squared"]         = d["house_age"] ** 2
    d["is_new"]              = int(d["house_age"] < 5)
    d["is_old"]              = int(d["house_age"] > 20)
    d["age_group"]           = float(min(int(d["house_age"] // 10), 3))
    d["quarter"]             = int(round((d["transaction_date"] % 1) * 4))
    d["is_recent"]           = int(d["transaction_date"] > median_date)
    d["price_zone_lat"]      = 2
    d["price_zone_lon"]      = 2

    c_scaled         = scaler_geo.transform([[d["latitude"], d["longitude"]]])
    d["geo_cluster"] = int(kmeans.predict(c_scaled)[0])

    cid = d["geo_cluster"]
    fallback = float(np.mean(list(price_dict.values())))
    d["cluster_avg_price"] = float(price_dict.get(cid, fallback))

    X_new = pd.DataFrame([d])[FEATURE_COLS]
    return float(model.predict(X_new)[0])