import pandas as pd
import streamlit as st

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00477/Real%20estate%20valuation%20data%20set.xlsx"
)

COLUMN_NAMES = [
    "No", "transaction_date", "house_age",
    "dist_mrt", "convenience_stores",
    "latitude", "longitude", "price_per_area",
]

FEATURE_META = {
    "transaction_date":   ("Quantitative", "Transaction date (e.g. 2013.25 = Q1 2013)"),
    "house_age":          ("Quantitative", "House age in years"),
    "dist_mrt":           ("Quantitative", "Distance to nearest MRT station (meters)"),
    "convenience_stores": ("Quantitative", "Number of nearby convenience stores"),
    "latitude":           ("Spatial",      "Geographic latitude"),
    "longitude":          ("Spatial",      "Geographic longitude"),
    "price_per_area":     ("Target",       "House price per unit area (10,000 TWD/ping)"),
}


@st.cache_data(show_spinner="Loading dataset …")
def load_raw_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_URL)
    df.columns = COLUMN_NAMES
    df.drop(columns=["No"], inplace=True)
    return df