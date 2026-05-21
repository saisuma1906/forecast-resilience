# -*- coding: utf-8 -*-
"""
Prepare Rossmann Pharmacy Store Sales data for Phase C & D pipeline.
Reads data/train.csv + data/store.csv
Writes outputs/processed_data.csv  (daily, Rossmann format)
Writes outputs/phase1_product_profiles.csv  (per-store baseline stats)
Run: python prepare_rossmann_data.py
"""
import pandas as pd
import numpy as np
import os, warnings
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = r"C:\Users\saisu\OneDrive - SUNY Polytechnic Institute\Desktop\forecast-resilience"
DATA_PATH   = os.path.join(ROOT, "data")
OUTPUT_PATH = os.path.join(ROOT, "outputs")

# ── Selected 19 stores (5 per type a/b/c/d, max open days, <5% zero sales) ──
SELECTED_STORES = [530, 310, 1099, 863, 453,   # type a (small)
                    85, 494,  733, 562, 423,     # type b (large)
                   680, 246,  197, 208, 217,     # type c (medium)
                   578, 299,  867,  54]          # type d (flagship)

print("=" * 60)
print("  Rossmann Data Preparation")
print("=" * 60)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("\n[1/4] Loading train.csv and store.csv...")
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), low_memory=False,
                    parse_dates=["Date"])
store = pd.read_csv(os.path.join(DATA_PATH, "store.csv"))
print("  train.csv:", train.shape, "| store.csv:", store.shape)

# ── 2. Filter to selected stores ──────────────────────────────────────────────
print("\n[2/4] Filtering to 19 selected stores...")
df = train[train["Store"].isin(SELECTED_STORES)].copy()
df = df[df["Open"] == 1].copy()          # exclude closed days
df = df[df["Sales"] > 0].copy()          # exclude zero-sales days
df = df.merge(store[["Store","StoreType","Assortment",
                      "CompetitionDistance","Promo2"]],
              on="Store", how="left")
df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
df = df.rename(columns={"Store": "id", "Sales": "sales"})
df = df.sort_values(["id", "date"]).reset_index(drop=True)

print("  Rows:", len(df), "| Stores:", df["id"].nunique())
print("  Date range:", df["date"].min(), "to", df["date"].max())
for sid in sorted(df["id"].unique()):
    n = len(df[df["id"]==sid])
    print(f"    Store {sid:4d}: {n} open days")

# ── 3. Save processed_data.csv ────────────────────────────────────────────────
print("\n[3/4] Saving processed_data.csv...")
out_cols = ["id", "date", "sales", "Promo", "StateHoliday",
            "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance"]
save_cols = [c for c in out_cols if c in df.columns]
df[save_cols].to_csv(os.path.join(OUTPUT_PATH, "processed_data.csv"), index=False)
print("  Saved:", os.path.join(OUTPUT_PATH, "processed_data.csv"))

# ── 4. Build phase1_product_profiles.csv with ARIMA baseline MAPSE ───────────
print("\n[4/4] Computing baseline ARIMA MAPSE per store (monthly)...")

# Aggregate to monthly
df["date_dt"] = pd.to_datetime(df["date"])
df["ym"]      = df["date_dt"].dt.to_period("M")
monthly = (df.groupby(["id","ym"])["sales"]
           .sum()
           .reset_index()
           .assign(date=lambda x: x["ym"].dt.to_timestamp())
           .sort_values(["id","date"]))

profiles = []
TEST_MONTHS = 6

for sid in sorted(monthly["id"].unique()):
    arr = monthly[monthly["id"]==sid].sort_values("date")["sales"].values.astype(float)
    if len(arr) < 18:
        print(f"    Store {sid}: too short ({len(arr)} months) -- skipping")
        continue

    test  = arr[-TEST_MONTHS:]
    train_arr = arr[:-TEST_MONTHS]
    denom = max(float(np.mean(arr)), 1.0)

    try:
        fc    = ARIMA(train_arr, order=(1,1,1)).fit().forecast(TEST_MONTHS)
        fc    = np.clip(fc, 0, None)
        mapse = float(np.mean(np.abs(test - fc)) / denom * 100)
    except Exception as e:
        print(f"    Store {sid}: ARIMA failed ({e}) -- using naive")
        fc    = np.full(TEST_MONTHS, float(np.mean(train_arr)))
        mapse = float(np.mean(np.abs(test - fc)) / denom * 100)

    store_row = store[store["Store"]==sid].iloc[0] if sid in store["Store"].values else {}
    profiles.append({
        "store_id":             sid,
        "store_type":           store_row.get("StoreType", "?") if isinstance(store_row, pd.Series) else "?",
        "assortment":           store_row.get("Assortment", "?") if isinstance(store_row, pd.Series) else "?",
        "competition_distance": store_row.get("CompetitionDistance", np.nan) if isinstance(store_row, pd.Series) else np.nan,
        "n_months":             len(arr),
        "mean_monthly_sales":   round(float(np.mean(arr)), 1),
        "std_monthly_sales":    round(float(np.std(arr)), 1),
        "cv":                   round(float(np.std(arr) / max(np.mean(arr), 1)), 3),
        "mapse_baseline":       round(mapse, 2),
    })
    print(f"    Store {sid:4d}: {len(arr)} months | ARIMA MAPSE = {mapse:.1f}%")

prof_df = pd.DataFrame(profiles)
prof_df.to_csv(os.path.join(OUTPUT_PATH, "phase1_product_profiles.csv"), index=False)
print("\n  Saved:", os.path.join(OUTPUT_PATH, "phase1_product_profiles.csv"))

print("\n" + "="*60)
print("  ROSSMANN DATA PREPARATION COMPLETE")
print("="*60)
print("  Stores prepared :", len(profiles))
print("  Avg baseline MAPSE:", round(prof_df["mapse_baseline"].mean(), 2), "%")
print("  Store types      :", dict(prof_df.groupby("store_type")["store_id"].count()))
print()
print("  Next steps:")
print("  1. python generate_phase_c_v3.py   (re-run Phase C)")
print("  2. python run_phase_d.py            (re-run Phase D)")
print("  3. python make_phase_d_nb.py        (rebuild notebooks)")
print("="*60)
