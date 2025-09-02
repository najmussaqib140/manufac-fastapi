import os
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Config
# -----------------------------------------------------------
CSV_URL = os.getenv("CSV_URL").strip()



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _to_date(s: Any) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)

def _validate_fuel(fuel: str) -> str:
    f = (fuel or "").strip().lower()
    if f not in ("petrol", "diesel"):
        raise HTTPException(status_code=400, detail="fuel must be 'petrol' or 'diesel'")
    return f

def _norm(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", str(s)).lower()
    return re.sub(r"\s+", " ", s).strip()

def _find_header(cols: List[str], want: str, allow_prefix: bool = True) -> Optional[str]:
    norm_map = {c: _norm(c) for c in cols}
    for orig, normed in norm_map.items():
        if normed == want:
            return orig
    if allow_prefix:
        for orig, normed in norm_map.items():
            if normed.startswith(want):
                return orig
    return None

def _maybe_convert_gdrive(url: str) -> str:
    m = re.search(r"https?://drive\.google\.com/file/d/([^/]+)/", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def _is_http_url(path: str) -> bool:
    return path.lower().startswith("http://") or path.lower().startswith("https://")



# ------------------------------------------------------------
class DataStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.mapping: Optional[Dict[str, str]] = None
        self.last_loaded: Optional[datetime] = None

    def load(self):
        #Choose source
        src = CSV_URL.strip()
        df = None

        try:
            if src and _is_http_url(src):
                # Handle Google Drive share link
                src = _maybe_convert_gdrive(src)
                df = pd.read_csv(src)
            elif src and os.path.exists(src):
                # Local absolute/relative file path
                df = pd.read_csv(src)
          
            else:
                raise FileNotFoundError(
                    f"No CSV source found. Checked CSV_URL='{CSV_URL}''."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to read CSV from '{src}'. "
                f"If using Google Drive, make sure the file is shared 'Anyone with the link' "
                f"and try the direct download link."
            ) from e

        print(">> CSV columns read:", df.columns.tolist())

        #  Detect headers robustly
        cols = df.columns.tolist()
        col_date  = _find_header(cols, "calendar day")
        col_city  = _find_header(cols, "metro cities")
        col_fuel  = _find_header(cols, "products")             # trailing space handled by _norm
        col_price = _find_header(cols, "retail selling price", allow_prefix=True)

        missing = [name for name, col in {
            "Calendar Day": col_date,
            "Metro Cities": col_city,
            "Products": col_fuel,
            "Retail Selling Price": col_price,
        }.items() if col is None]

        if missing:
            raise ValueError(
                f"Required headers not found: {missing}. "
                f"Found columns: {cols}"
            )

        #  Normalize column names
        df = df.rename(columns={
            col_date: "date",
            col_city: "city",
            col_fuel: "fuel",
            col_price: "price",
        })

        #Clean/parse
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["city"] = df["city"].astype(str).str.strip()
        df["fuel"] = df["fuel"].astype(str).str.lower().str.strip()
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

        # Drop invalid dates
        df = df.dropna(subset=["date"])

        # Pivot to wide (petrol, diesel columns)
        df_pivot = (
            df.pivot_table(
                index=["date", "city"],
                columns="fuel",
                values="price",
                aggfunc="first"
            )
            .reset_index()
            .rename(columns={"petrol": "petrol", "diesel": "diesel"})
        )

        # Ensure both fuel columns exist
        if "petrol" not in df_pivot.columns:
            df_pivot["petrol"] = 0.0
        if "diesel" not in df_pivot.columns:
            df_pivot["diesel"] = 0.0

        df_pivot = (
            df_pivot
            .fillna({"petrol": 0.0, "diesel": 0.0})
            .sort_values(by=["city", "date"])
            .reset_index(drop=True)
        )

        self.df = df_pivot
        self.mapping = {"date": "date", "city": "city", "petrol": "petrol", "diesel": "diesel"}
        self.last_loaded = datetime.utcnow()

    def ensure_loaded(self):
        if self.df is None:
            self.load()

    # ---------- analytics helpers ----------
    def cities(self) -> List[str]:
        self.ensure_loaded()
        return sorted(self.df[self.mapping["city"]].dropna().unique().tolist())

    def latest_for_city(self, city: str) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        dfc = self.df[self.df[m["city"]].str.lower() == city.lower()]
        if dfc.empty:
            raise HTTPException(404, detail=f"City '{city}' not found")
        row = dfc.iloc[-1]
        return {
            "city": row[m["city"]],
            "date": row[m["date"]].date().isoformat(),
            "petrol": float(row[m["petrol"]]),
            "diesel": float(row[m["diesel"]]),
        }

    def price_on(self, city: str, fuel: str, date: datetime) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        dfc = self.df[self.df[m["city"]].str.lower() == city.lower()]
        if dfc.empty:
            raise HTTPException(404, detail=f"City '{city}' not found")

        df_day = dfc[dfc[m["date"]] == pd.Timestamp(date.date())]
        if df_day.empty:
            raise HTTPException(404, detail=f"No data for {city} on {date.date().isoformat()}")

        col = m["petrol"] if fuel == "petrol" else m["diesel"]
        return {"city": city, "fuel": fuel, "date": date.date().isoformat(), "price": float(df_day.iloc[0][col])}

    def series_range(self, city: str, fuel: str, start: datetime, end: datetime) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        dfc = self.df[self.df[m["city"]].str.lower() == city.lower()]
        if dfc.empty:
            raise HTTPException(404, detail=f"City '{city}' not found")
        if end < start:
            raise HTTPException(400, detail="end must be >= start")

        mask = (dfc[m["date"]] >= pd.Timestamp(start.date())) & (dfc[m["date"]] <= pd.Timestamp(end.date()))
        dfr = dfc.loc[mask]
        col = m["petrol"] if fuel == "petrol" else m["diesel"]

        result = [{"date": d.date().isoformat(), "price": float(v)} for d, v in zip(dfr[m["date"]], dfr[col])]
        return {"city": city, "fuel": fuel, "start": start.date().isoformat(), "end": end.date().isoformat(), "data": result}

    def summary_city(self, city: str) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        dfc = self.df[self.df[m["city"]].str.lower() == city.lower()]
        if dfc.empty:
            raise HTTPException(404, detail=f"City '{city}' not found")

        latest = dfc.iloc[-1]

        def stats(col):
            s = dfc[col]
            return {"min": float(s.min()), "max": float(s.max()), "avg": float(s.mean()), "latest": float(latest[col])}

        return {
            "city": latest[m["city"]],
            "date_latest": latest[m["date"]].date().isoformat(),
            "petrol": stats(m["petrol"]),
            "diesel": stats(m["diesel"]),
        }

    def compare_spread(self, city: str, start: datetime, end: datetime) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        dfc = self.df[self.df[m["city"]].str.lower() == city.lower()]
        if dfc.empty:
            raise HTTPException(404, detail=f"City '{city}' not found")
        if end < start:
            raise HTTPException(400, detail="end must be >= start")

        mask = (dfc[m["date"]] >= pd.Timestamp(start.date())) & (dfc[m["date"]] <= pd.Timestamp(end.date()))
        dfr = dfc.loc[mask].copy()
        dfr["spread"] = dfr[m["petrol"]] - dfr[m["diesel"]]
        series = [{"date": d.date().isoformat(), "spread": float(s)} for d, s in zip(dfr[m["date"]], dfr["spread"])]

        return {
            "city": city,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "avg_spread": float(dfr["spread"].mean()) if not dfr.empty else 0.0,
            "data": series,
        }

    def top_increase(self, fuel: str, days: int, n: int) -> Dict[str, Any]:
        self.ensure_loaded()
        m = self.mapping
        end = self.df[m["date"]].max()
        start = end - pd.Timedelta(days=days)
        col = m["petrol"] if fuel == "petrol" else m["diesel"]

        dfw = self.df[(self.df[m["date"]] >= start) & (self.df[m["date"]] <= end)].copy()
        if dfw.empty:
            return {"fuel": fuel, "days": days, "as_of": pd.Timestamp(end).date().isoformat(), "top": []}

        dfw["pct_change"] = dfw.groupby(m["city"])[col].pct_change() * 100.0
        agg = dfw.groupby(m["city"])["pct_change"].max().fillna(0.0).sort_values(ascending=False)
        out = [{"city": c, "max_pct_increase": float(v)} for c, v in agg.head(n).items()]
        return {"fuel": fuel, "days": days, "as_of": pd.Timestamp(end).date().isoformat(), "top": out}


store = DataStore()

app = FastAPI(
    title="RSP Petrol/Diesel Analytics API",
    version="1.0.0",
    description="FastAPI service for in-memory analytics on Petrol/Diesel RSP dataset (no API models)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Routes -----------------------------
@app.get("/health", tags=["meta"])
def health():
    ts = store.last_loaded.isoformat() + "Z" if store.last_loaded else None
    return {"status": "ok", "last_loaded_utc": ts}

@app.get("/cities", tags=["catalog"])
def get_cities():
    return store.cities()

@app.get("/latest", tags=["analytics"])
def latest(city: str = Query(..., description="City name (case-insensitive)")):
    return store.latest_for_city(city)

@app.get("/price", tags=["analytics"])
def price(
    city: str = Query(..., description="City name (case-insensitive)"),
    fuel: str = Query(..., description="petrol or diesel"),
    date: datetime = Query(..., description="ISO date e.g. 2024-01-31"),
):
    fuel = _validate_fuel(fuel)
    return store.price_on(city, fuel, date)

@app.get("/range", tags=["analytics"])
def series(
    city: str = Query(...),
    fuel: str = Query(...),
    start: datetime = Query(..., description="ISO date start"),
    end: datetime = Query(..., description="ISO date end"),
):
    fuel = _validate_fuel(fuel)
    return store.series_range(city, fuel, start, end)

@app.get("/summary", tags=["analytics"])
def summary(city: str = Query(...)):
    return store.summary_city(city)

@app.get("/compare", tags=["analytics"])
def compare(
    city: str = Query(...),
    start: datetime = Query(...),
    end: datetime = Query(...),
):
    return store.compare_spread(city, start, end)

@app.get("/top_increase", tags=["analytics"])
def top_increase(
    fuel: str = Query(..., description="petrol or diesel"),
    days: int = Query(30, ge=1, le=365),
    n: int = Query(5, ge=1, le=50),
):
    fuel = _validate_fuel(fuel)
    return store.top_increase(fuel, days, n)



#Testing-----------------------------------
# Cities list
# http://127.0.0.1:8000/cities

# Latest for a city (example: Delhi)
# http://127.0.0.1:8000/latest?city=Delhi

# Price on date (petrol @ Delhi on 2025-06-20)
# http://127.0.0.1:8000/price?city=Delhi&fuel=petrol&date=2025-06-20

# Range series (diesel @ Delhi, 2025-06-17 → 2025-06-20)
# http://127.0.0.1:8000/range?city=Delhi&fuel=diesel&start=2025-06-17&end=2025-06-20

# City summary (Mumbai)
# http://127.0.0.1:8000/summary?city=Mumbai

# Compare petrol-diesel spread (Chennai, 2025-06-17 → 2025-06-20)
# http://127.0.0.1:8000/compare?city=Chennai&start=2025-06-17&end=2025-06-20

# Top increase (petrol, last 30 days, top 5)
# http://127.0.0.1:8000/top_increase?fuel=petrol&days=30&n=5