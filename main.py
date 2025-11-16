import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# SQLite via SQLAlchemy (lightweight metadata cache)
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# NOTE: We avoid importing heavy deps (fastf1/matplotlib/scipy) at module load
# so the API can start even if those optional deps fail to install.
# We import them lazily inside endpoint handlers.

app = FastAPI(title="F1 Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite DB (file on disk for quick metadata caching)
DB_PATH = os.getenv("SQLITE_PATH", "/tmp/f1-analytics.db")
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)


# ---------- Schemas ----------
class DriverLapTime(BaseModel):
    driver: str
    lap_number: int
    lap_time_ms: Optional[float]


class TelemetryPoint(BaseModel):
    time_s: float
    speed: Optional[float] = None
    throttle: Optional[float] = None
    brake: Optional[float] = None
    gear: Optional[int] = None
    drs: Optional[int] = None


# ---------- Helpers ----------

def _get_fastf1():
    """Lazy import FastF1 and set up cache. Raise 503 if unavailable."""
    try:
        import fastf1 as ff1  # type: ignore
        # Initialize FastF1 cache directory
        cache_dir = os.getenv("FASTF1_CACHE", "/tmp/fastf1-cache")
        os.makedirs(cache_dir, exist_ok=True)
        ff1.Cache.enable_cache(cache_dir)
        return ff1
    except ModuleNotFoundError as e:
        # Surface a clear server-available but feature-unavailable error
        raise HTTPException(status_code=503, detail=f"FastF1 dependency missing: {e}. Try reinstalling dependencies.")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"FastF1 initialization failed: {e}")


def _load_session(year: int, gp: int | str, session: str):
    ff1 = _get_fastf1()
    try:
        ses = ff1.get_session(year, gp, session)
        ses.load(laps=True, telemetry=True, weather=True)
        return ses
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load session: {e}")


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "F1 Analytics API is running"}


@app.get("/api/seasons/{year}/events")
def list_events(year: int):
    ff1 = _get_fastf1()
    try:
        ev = ff1.get_event_schedule(year, include_testing=False)
        items = []
        for _, row in ev.iterrows():
            items.append({
                "round": int(row.get("RoundNumber") or row.get("round", 0)),
                "event_name": row.get("EventName") or row.get("EventFormat"),
                "country": row.get("Country"),
                "location": row.get("Location"),
                "event_date": str(row.get("EventDate"))
            })
        return {"year": year, "events": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/seasons/{year}/races/{round}/results")
def race_results(year: int, round: int):
    session = _load_session(year, round, "R")
    results = session.results
    if results is None or results.empty:
        raise HTTPException(status_code=404, detail="No results available")

    out = []
    for _, r in results.iterrows():
        out.append({
            "position": int(r.get("Position", r.get("Pos", 0)) or 0),
            "driver": str(r.get("Driver", r.get("DriverNumber", ""))),
            "driver_number": int(r.get("DriverNumber", 0) or 0),
            "team": r.get("TeamName") or r.get("Team", ""),
            "status": r.get("Status", ""),
            "points": float(r.get("Points", 0) or 0),
            "time": str(r.get("Time", "")),
        })
    return {"year": year, "round": round, "results": out}


@app.get("/api/seasons/{year}/races/{round}/laps")
def driver_lap_times(year: int, round: int, driver: Optional[str] = Query(None)):
    import pandas as pd  # local import to avoid startup failures
    session = _load_session(year, round, "R")
    laps = session.laps
    if laps is None or laps.empty:
        raise HTTPException(status_code=404, detail="No laps available")

    if driver:
        laps = laps.pick_driver(driver)

    laps = laps[['Driver', 'LapNumber', 'LapTime']].copy()
    laps['LapTime_ms'] = laps['LapTime'].dt.total_seconds() * 1000

    out = [
        {
            "driver": row['Driver'],
            "lap_number": int(row['LapNumber']),
            "lap_time_ms": float(row['LapTime_ms']) if pd.notna(row['LapTime_ms']) else None
        }
        for _, row in laps.iterrows()
    ]

    return {"year": year, "round": round, "laps": out}


@app.get("/api/seasons/{year}/races/{round}/telemetry")
def driver_telemetry(
    year: int,
    round: int,
    driver: str = Query(..., description="Driver code, e.g., VER, HAM, LEC"),
    lap: Optional[int] = Query(None, description="Specific lap number; if omitted, use fastest lap")
):
    import pandas as pd
    session = _load_session(year, round, "R")
    laps = session.laps.pick_driver(driver)
    if laps.empty:
        raise HTTPException(status_code=404, detail="No laps for driver")

    if lap is None:
        lap_obj = laps.pick_fastest()
    else:
        lap_obj = laps.loc[laps['LapNumber'] == lap]
        if lap_obj.empty:
            raise HTTPException(status_code=404, detail="Lap not found")
        lap_obj = lap_obj.iloc[0]

    tel = lap_obj.get_telemetry()
    tel['Time_s'] = tel['Time'].dt.total_seconds()

    def safe(v):
        return float(v) if pd.notna(v) else None

    out = [
        {
            "time_s": safe(row['Time_s']),
            "speed": safe(row.get('Speed')),
            "throttle": safe(row.get('Throttle')),
            "brake": safe(row.get('Brake')),
            "gear": int(row.get('nGear')) if pd.notna(row.get('nGear')) else None,
            "drs": int(row.get('DRS')) if pd.notna(row.get('DRS')) else None,
        }
        for _, row in tel.iterrows()
    ]

    return {"year": year, "round": round, "driver": driver, "lap": int(lap_obj['LapNumber']), "telemetry": out}


@app.get("/api/seasons/{year}/races/{round}/drivers")
def race_drivers(year: int, round: int):
    session = _load_session(year, round, "R")
    laps = session.laps
    drivers = sorted(list(set(laps['Driver'].dropna().unique().tolist())))
    numbers = (
        laps[['Driver', 'DriverNumber']]
        .dropna()
        .drop_duplicates()
        .set_index('Driver')['DriverNumber']
        .to_dict()
    )
    return {"year": year, "round": round, "drivers": drivers, "numbers": numbers}


@app.get("/api/seasons/{year}/races/{round}/position-chart")
def position_chart(year: int, round: int):
    import pandas as pd
    session = _load_session(year, round, "R")
    laps = session.laps
    if laps.empty:
        raise HTTPException(status_code=404, detail="No laps available")

    per_driver = {}
    for drv in laps['Driver'].dropna().unique():
        dl = laps.pick_driver(drv)
        per_driver[drv] = [
            {
                "lap": int(row['LapNumber']),
                "position": int(row['Position']) if pd.notna(row['Position']) else None,
            }
            for _, row in dl[['LapNumber', 'Position']].iterrows()
        ]

    return {"year": year, "round": round, "positions": per_driver}


@app.get("/api/drivers/{driver}/profile")
def driver_profile(driver: str, year: Optional[int] = None):
    import pandas as pd
    year = year or 2023
    session = _load_session(year, 1, "R")
    laps = session.laps
    if laps.empty:
        raise HTTPException(status_code=404, detail="No data")

    dl = laps.pick_driver(driver)
    if dl.empty:
        raise HTTPException(status_code=404, detail="Driver not found in season")

    team = (
        dl[['Team']]
        .dropna()
        .iloc[0]['Team']
    ) if not dl[['Team']].dropna().empty else None

    number = (
        dl[['DriverNumber']]
        .dropna()
        .iloc[0]['DriverNumber']
    ) if not dl[['DriverNumber']].dropna().empty else None

    return {
        "driver": driver,
        "year": year,
        "team": str(team) if team is not None else None,
        "number": int(number) if pd.notna(number) else None,
    }


# Health check that never touches FastF1
@app.get("/test")
def test():
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)"))
            conn.execute(text("INSERT OR REPLACE INTO meta (k, v) VALUES ('status', 'ok')"))
            res = conn.execute(text("SELECT v FROM meta WHERE k='status'"))
            ok = res.scalar_one_or_none()
    except SQLAlchemyError as e:
        return {"backend": "running", "sqlite": f"error: {e}"}

    # Probe availability of FastF1 without crashing
    try:
        _get_fastf1()
        fastf1_ready = True
    except HTTPException:
        fastf1_ready = False

    return {"backend": "running", "sqlite": ok, "fastf1": "ready" if fastf1_ready else "unavailable"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
