import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_TOP = 10  # número de activos candidatos a devolver (ajustable)
MIN_HOURS_THRESHOLD = 24 * 7  # al menos 1 semana de horas válidas

# parámetros de scoring
W_VOL = 0.45
W_STATIC = 0.20
W_GAPS = 0.20
W_NEG_RET = 0.15


def score_file(path):
    """
    Lee el CSV y devuelve un diccionario con métricas y score.
    Asume formato CryptoDataDownload: columnas incluyen date, open, high, low, close, Volume USD (o Volume USD)
    """
    nombre = os.path.basename(path)
    try:
        df = pd.read_csv(path, skiprows=1)
    except Exception as e:
        return {"file": path, "ok": False, "error": f"read_error:{e}"}

    # detectar columnas de volumen en USD (nombres comunes)
    vol_cols = [c for c in df.columns if "Volume" in c and "USD" in c]
    vol_col = vol_cols[0] if vol_cols else (df.columns[-1] if len(df.columns) >= 8 else None)

    required = ["date", "open", "high", "low", "close"]
    if not all(col in df.columns for col in required):
        return {"file": path, "ok": False, "error": "missing_columns"}

    # parse date
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except:
        return {"file": path, "ok": False, "error": "date_parse"}

    df = df.set_index("date").sort_index()
    # keep relevant cols
    cols_keep = ["open", "high", "low", "close"]
    if vol_col is not None:
        cols_keep.append(vol_col)
        df = df[cols_keep]
        df = df.rename(columns={vol_col: "vol_usd"})
    else:
        df = df[cols_keep]
        df["vol_usd"] = np.nan

    # coercions
    for c in ["open", "high", "low", "close", "vol_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic filters
    total_hours = len(df)
    if total_hours < MIN_HOURS_THRESHOLD:
        return {"file": path, "ok": False, "error": "too_few_hours", "hours": total_hours}

    # percent zero volume (or missing)
    vol_zero = (df["vol_usd"].fillna(0) <= 0).sum()
    pct_vol_zero = vol_zero / total_hours

    # percent static price (open==high==low==close)
    static = ((df["open"] == df["high"]) & (df["open"] == df["low"]) & (df["open"] == df["close"]))
    pct_static = static.sum() / total_hours

    # gaps: expected hourly frequency
    expected = pd.date_range(df.index.min(), df.index.max(), freq="H")
    gaps = len(expected) - df.index.nunique()
    pct_gaps = gaps / len(expected) if len(expected) > 0 else 1.0

    # returns and negative returns %
    ret = df["close"].pct_change().dropna()
    pct_neg = (ret < 0).sum() / len(ret) if len(ret) > 0 else 1.0
    mean_vol = df["vol_usd"].replace(0, np.nan).dropna().mean() if df["vol_usd"].notna().any() else 0.0

    # Normalize metrics to [0,1] for scoring (higher is better)
    # We want high liquidity -> high mean_vol (scale by log)
    vol_score = np.tanh(np.log1p(mean_vol) / 10.0) if mean_vol > 0 else 0.0
    # low pct zero -> better
    zero_score = 1.0 - min(1.0, pct_vol_zero * 2.0)
    static_score = 1.0 - min(1.0, pct_static * 2.0)
    gaps_score = 1.0 - min(1.0, pct_gaps * 5.0)
    neg_score = 1.0 - min(1.0, pct_neg * 2.0)

    score = (
        W_VOL * vol_score +
        W_STATIC * static_score +
        W_GAPS * gaps_score +
        W_NEG_RET * neg_score
    )

    return {
        "file": path,
        "ok": True,
        "hours": total_hours,
        "mean_vol": float(mean_vol) if not np.isnan(mean_vol) else 0.0,
        "pct_vol_zero": float(pct_vol_zero),
        "pct_static": float(pct_static),
        "pct_gaps": float(pct_gaps),
        "pct_neg_ret": float(pct_neg),
        "score": float(score)
    }


def curate_universe(data_dir=DATA_DIR, top_n=OUTPUT_TOP):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_1h.csv")]
    results = []
    for f in files:
        res = score_file(f)
        if res.get("ok"):
            results.append(res)
    if not results:
        raise RuntimeError("No valid crypto files found in data directory")

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    top = df.head(top_n)

    ARCHIVOS = [r["file"] for _, r in top.iterrows()]

    # Print top 10 summary
    print("=== TOP CANDIDATES ===")
    print(top[["file", "hours", "mean_vol", "pct_vol_zero", "pct_static", "pct_gaps", "pct_neg_ret", "score"]].head(20).to_string(index=False))

    # Produce ARCHIVOS list ready to paste
    print("\nCopia y pega esta lista en tu configuración:\n")
    print("ARCHIVOS = [")
    for a in ARCHIVOS[:50]:
        print(f"    os.path.join(DATA_DIR, \"{os.path.basename(a)}\"),")
    print("]\n")

    return ARCHIVOS


if __name__ == "__main__":
    curate_universe()
