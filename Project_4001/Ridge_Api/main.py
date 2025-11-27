# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Thai Rubber Price MLaaS",
    description=(
        "Predict thai_price given year and month. "
        "If the month is in history, return actual + prediction; "
        "if it's in the future, reuse last known external factors to predict."
    ),
    version="1.1.0"
)

# =========================
# 1. Load model artifacts
# =========================
artifacts = joblib.load("ridge_trp_model.joblib")
model = artifacts["model"]
scaler = artifacts["scaler"]
external_factors = artifacts["features"]  # list of feature column names

# =========================
# 2. Load preprocessed DataFrame
# =========================
# This file should be created in your notebook with:
# df.to_parquet("processed_df.parquet")
try:
    df = pd.read_parquet("processed_df.parquet")
except FileNotFoundError:
    raise RuntimeError(
        "processed_df.parquet not found. "
        "Make sure you saved it in your training notebook and copied it into this folder."
    )

# If df was saved with index as date_month, we ensure it's a DateTimeIndex.
# If df was saved with a column 'date_month', we convert & set it as index.
if isinstance(df.index, pd.RangeIndex) or df.index.dtype == "object":
    if "date_month" in df.columns:
        df["date_month"] = pd.to_datetime(df["date_month"])
        df = df.set_index("date_month")
    else:
        raise RuntimeError(
            "DataFrame index is not a datetime index and no 'date_month' column found. "
            "Check how you saved processed_df."
        )

# Ensure datetime index and monthly frequency
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.asfreq("MS")  # Monthly Start

min_date = df.index.min()
max_date = df.index.max()

# =========================
# 3. Sanity check columns
# =========================
required_cols = ["thai_price"] + list(external_factors)
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns in processed_df: {missing}")


# =========================
# 4. Request / Response models
# =========================
class PriceRequest(BaseModel):
    year: int
    month: int  # 1-12


class PriceResponse(BaseModel):
    year: int
    month: int
    date: str
    is_future: bool
    has_actual: bool
    actual_price: float | None
    predicted_price: float
    feature_source_date: str
    unit: str = "THB/kg"


# =========================
# 5. API endpoints
# =========================
@app.post("/predict", response_model=PriceResponse)
def predict_price(req: PriceRequest):
    # 1) Validate and construct date
    try:
        date = pd.Timestamp(year=req.year, month=req.month, day=1)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid year or month. Month must be between 1 and 12."
        )

    # If requested date is before the dataset starts -> error
    if date < min_date:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Requested date {date.strftime('%Y-%m')} is before available data "
                f"({min_date.strftime('%Y-%m')})."
            )
        )

    # Determine if this is historical (in df) or future (> max_date)
    is_future = date > max_date

    if not is_future:
        # Historical or current data: use the row for that month
        if date not in df.index:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No data row in processed_df for {date.strftime('%Y-%m')}. "
                    f"Check if this month exists in your processed dataframe."
                )
            )
        row = df.loc[date]
        feature_source_date = date
        actual_val = row.get("thai_price")
        has_actual = not pd.isna(actual_val)
        actual_price = float(actual_val) if has_actual else None
    else:
        # Future date: reuse last available external factors (from max_date)
        feature_source_date = max_date
        row = df.loc[max_date]
        has_actual = False
        actual_price = None

    # 2) Prepare feature matrix
    X_raw = row[external_factors].to_frame().T  # shape (1, n_features)

    if X_raw.isna().any().any():
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature values contain NaN for source date "
                f"{feature_source_date.strftime('%Y-%m')}. Check preprocessing."
            )
        )

    X_scaled = scaler.transform(X_raw)

    # 3) Prediction
    pred_price = float(model.predict(X_scaled)[0])

    return PriceResponse(
        year=req.year,
        month=req.month,
        date=date.strftime("%Y-%m-%d"),
        is_future=is_future,
        has_actual=has_actual,
        actual_price=actual_price,
        predicted_price=pred_price,
        feature_source_date=feature_source_date.strftime("%Y-%m-%d"),
    )


@app.get("/")
def root():
    return {
        "message": "Thai Rubber Price MLaaS is running.",
        "usage": (
            "POST /predict with JSON: {\"year\": 2025, \"month\": 1}. "
            "For future months, external factors are assumed equal to the last observed month."
        ),
        "data_range": {
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
        },
    }
