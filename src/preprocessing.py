import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Cleaning data …")
def clean_data(_df: pd.DataFrame):
    df = _df.copy()
    log = []

    # Step 1 – Outlier removal
    before = len(df)
    Q1 = df["price_per_area"].quantile(0.01)
    Q3 = df["price_per_area"].quantile(0.99)
    df = df[(df["price_per_area"] >= Q1) & (df["price_per_area"] <= Q3)].reset_index(drop=True)
    removed = before - len(df)
    log.append({
        "Step": "Outlier Removal (1st–99th pct)",
        "Before": before,
        "After": len(df),
        "Removed": removed,
        "Note": f"Price kept between {Q1:.1f} – {Q3:.1f}",
    })

    # Step 2 – Duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.append({
        "Step": "Duplicate Removal",
        "Before": before,
        "After": len(df),
        "Removed": before - len(df),
        "Note": "Exact row duplicates dropped",
    })

    # Step 3 – Null audit
    nulls = int(df.isnull().sum().sum())
    log.append({
        "Step": "Null Audit",
        "Before": len(df),
        "After": len(df),
        "Removed": 0,
        "Note": f"{nulls} null value(s) found in raw data",
    })

    return df, log