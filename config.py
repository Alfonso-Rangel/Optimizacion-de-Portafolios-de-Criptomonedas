# config.py

import os
from datetime import datetime

# --- Paths ---
DATA_DIR = "data"
T_BILL_FILE = os.path.join(DATA_DIR, "DTB4WK.csv")

# ================ Config ================
ARCHIVOS = [
    os.path.join(DATA_DIR, "Gemini_BTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_ETHUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_DOGEUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_LINKUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_LTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_ANKRUSD_1h.csv"),
    #os.path.join(DATA_DIR, "Gemini_APEUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_BATUSD_1h.csv"),
    #os.path.join(DATA_DIR, "Gemini_BTCGUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_COMPUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_CRVUSD_1h.csv"),
    #os.path.join(DATA_DIR, "Gemini_CUBEUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_FETUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_GRTUSD_1h.csv"),
    #os.path.join(DATA_DIR, "Gemini_MASKUSD_1h.csv"),
    #os.path.join(DATA_DIR, "Gemini_SHIBUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_UMAUSD_1h.csv"),
]

FECHA_INICIO_SIMULACION = "2024-01-01"
FECHA_FIN_SIMULACION = "2025-11-02" 

# Ventana de entrenamiento (Lookback Window)
VENTANA_HORAS = 4 * 7 * 24  # 672 

# Paso de rebalanceo (Rebalancing Frequency)
PASO_HORAS = 7 * 24  # 168