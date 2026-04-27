"""
PASUL 1: DEFINIȚIA FEATURES
"""

import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "BridgeAI_Dataset", "Data Cleaner", "Clean.db")

FEATURE_SCHEMAS = {
    "laptops": [
        {"col": "price",              "direction": "lower_better",  "weight": 1.3},
        {"col": "capacity",           "direction": "higher_better", "weight": 1.5},
        {"col": "capacitatessd",      "direction": "higher_better", "weight": 1.4},
        {"col": "frecventaturbomax",  "direction": "higher_better", "weight": 1.3},
        {"col": "numarnuclee",        "direction": "higher_better", "weight": 1.3},
        {"col": "ratarefresh",        "direction": "higher_better", "weight": 1.4},
        {"col": "memoriededicata",    "direction": "higher_better", "weight": 1.5},
        {"col": "weight",             "direction": "lower_better",  "weight": 0.8},
        {"col": "display_size",       "direction": "neutral",       "weight": 0.7},
        {"col": "placavideo",         "direction": "categorical",   "weight": 1.5},
        {"col": "category",           "direction": "categorical",   "weight": 1.2},
        {"col": "sistemdeoperare",    "direction": "categorical",   "weight": 1.1},
    ],
    "smartphones": [
        {"col": "price",                  "direction": "lower_better",  "weight": 1.2},
        {"col": "ram",                    "direction": "higher_better", "weight": 1.4},
        {"col": "memorieinternaflash",    "direction": "higher_better", "weight": 1.3},
        {"col": "capacitateacumulator",   "direction": "higher_better", "weight": 1.5},
        {"col": "camerafotoprincipala",   "direction": "higher_better", "weight": 1.0},
        {"col": "weight",                 "direction": "lower_better",  "weight": 0.7},
        {"col": "highrefreshrate",        "direction": "higher_better", "weight": 1.3},
        {"col": "marimedisplay",          "direction": "neutral",       "weight": 0.8},
    ],
    "monitors": [
        {"col": "price",              "direction": "lower_better",  "weight": 1.2},
        {"col": "display_size",       "direction": "higher_better", "weight": 1.5},
        {"col": "refreshrate",        "direction": "higher_better", "weight": 1.5},
        {"col": "timpderaspuns",      "direction": "lower_better",  "weight": 1.1},
        {"col": "brightness",         "direction": "higher_better", "weight": 1.3},
        {"col": "recomandatpentru",   "direction": "categorical",   "weight": 1.4},
        {"col": "tippanel",           "direction": "categorical",   "weight": 1.4},
    ],
    "tvs": [
        {"col": "price",              "direction": "lower_better",  "weight": 1.2},
        {"col": "diagonalainch",      "direction": "higher_better", "weight": 1.5},
        {"col": "iesiresunetrms",     "direction": "higher_better", "weight": 1.3},
        {"col": "smarttv",            "direction": "categorical",   "weight": 1.3},
        {"col": "imaginehd",          "direction": "categorical",   "weight": 1.4},
    ],
    "headphones": [
        {"col": "price",              "direction": "lower_better",  "weight": 1.2},
        {"col": "battery_life_music", "direction": "higher_better", "weight": 1.3},
        {"col": "noise_canceling",    "direction": "categorical",   "weight": 1.4},
        {"col": "type",               "direction": "categorical",   "weight": 1.1},
        {"col": "headphone_type",     "direction": "categorical",   "weight": 0.9},
        {"col": "microphone",         "direction": "categorical",   "weight": 1.3},
    ],
    "gaming_headphones": [
        {"col": "price",              "direction": "lower_better",  "weight": 1.2},
        {"col": "battery_life_music", "direction": "higher_better", "weight": 1.1},
        {"col": "noise_canceling",    "direction": "categorical",   "weight": 1.1},
        {"col": "type",               "direction": "categorical",   "weight": 1.1},
        {"col": "microphone",         "direction": "categorical",   "weight": 1.5},
    ],
}

CATEGORICAL_MAPS = {
    "placavideo": {"Dedicata": 1.0, "Integrata": 0.3, "AMD Radeon": 0.85, "NVIDIA": 0.95, "Intel": 0.5},
    "category": { 
        "Gaming": 1.0, 
        "Workstation": 0.95, 
        "Grafica": 0.95,
        "Multimedia": 0.75, 
        "Business": 0.65, 
        "Office": 0.65,
        "Ultraportabil": 0.6,
        "Casual": 0.5,
        "Creator": 0.95,
        "Studio": 0.95,
    },
    "sistemdeoperare": {
        "Windows 11 Pro": 0.95, 
        "Windows 11 Home": 0.85, 
        "Windows 11": 0.85,
        "Windows 10": 0.75, 
        "Windows 10 Pro": 0.80,
        "macOS": 0.95, 
        "macOS Sonoma": 0.95,
        "Linux": 0.85, 
        "Ubuntu": 0.85,
        "Fara sistem de operare": 0.2, 
        "FreeDOS": 0.2,
        "N/A": 0.3,
    },
    "recomandatpentru": {
        "Gaming": 1.0, 
        "Grafica": 0.95, 
        "Fotografie": 0.90,
        "Multimedia": 0.8,
        "Office": 0.7, 
        "Birou": 0.7,
        "Videoconferinta": 0.65, 
        "Design": 0.95,
        "Digital Signage": 0.5,
        "Creativitate": 0.95,
    },
    "tippanel": {
        "IPS": 0.85, 
        "OLED": 1.0, 
        "VA": 0.75, 
        "QLED": 0.95,
        "Mini-LED": 0.95,
        "TN": 0.5,
        "LCD": 0.7,
        "LED": 0.7,
    },
    "smarttv": {"yes": 1.0, "no": 0.3, "1": 1.0, "0": 0.3, "True": 1.0, "False": 0.3},
    "imaginehd": {
        "4K Ultra HD": 0.85, 
        "Full HD": 0.65, 
        "HD Ready": 0.4, 
        "8K": 1.0,
        "2K": 0.75,
        "1080p": 0.65,
        "720p": 0.4,
    },
    "noise_canceling": {
        "yes": 1.0, 
        "no": 0.2, 
        "1": 1.0, 
        "0": 0.2, 
        "Active": 1.0,
        "Passive": 0.4,
        "Hybrid": 0.95,
        "True": 1.0,
        "False": 0.2,
    },
    "type": {
        "Wireless": 1.0, 
        "True Wireless": 1.0,
        "Wired": 0.4, 
        "Bluetooth": 1.0,
        "2.4GHz": 0.9,
        "USB": 0.3,
    },
    "headphone_type": { 
        "Over-ear": 0.9, 
        "Circumaural": 0.9,
        "On-ear": 0.75, 
        "Supraaural": 0.75,
        "In-ear": 0.7,
        "Earbud": 0.65,
        "Open-back": 0.85,
        "Closed-back": 0.8,
    },
    "microphone": {
        "yes": 1.0, 
        "no": 0.2, 
        "1": 1.0, 
        "0": 0.2, 
        "built-in": 1.0,
        "detachable": 0.8,
        "True": 1.0,
        "False": 0.2,
    },
}

def load_table(table_name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    conn.close()
    df.replace(-1, np.nan, inplace=True)
    df.replace("-1", np.nan, inplace=True)
    
    return df

def encode_categorical(df, col):
    mapping = CATEGORICAL_MAPS.get(col, {})
    if not mapping:
        return pd.Series([0.5] * len(df))
    
    return df[col].apply(lambda v: mapping.get(str(v).strip(), 0.5) if pd.notna(v) else 0.5)

def normalize_numeric(series, direction):
    series = pd.to_numeric(series, errors='coerce')
    
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series))
    
    normalized = (series - min_val) / (max_val - min_val)
    normalized = normalized.fillna(0.5)
    
    if direction == "lower_better":
        normalized = 1.0 - normalized
    elif direction == "neutral":
        mean_norm = normalized.mean()
        normalized = 1.0 - abs(normalized - mean_norm)
    
    return normalized

def build_feature_matrix(df, table_name):
    schema = FEATURE_SCHEMAS.get(table_name, [])
    feature_df = pd.DataFrame(index=df.index)
    
    for feature_def in schema:
        col = feature_def["col"]
        direction = feature_def["direction"]
        weight = feature_def["weight"]
        
        if col not in df.columns:
            continue
        
        if direction == "categorical":
            encoded = encode_categorical(df, col)
        else:
            encoded = normalize_numeric(df[col], direction)
        feature_df[f"feat_{col}"] = encoded * weight
    
    return feature_df

if __name__ == "__main__":
    print("=== TEST FEATURES.PY ===")
    df = load_table("laptops")
    print(f"Laptops loaded: {len(df)} rows")
    
    feat_matrix = build_feature_matrix(df, "laptops")
    print(f"Feature matrix shape: {feat_matrix.shape}")
    print(f"Features: {list(feat_matrix.columns)}")
    print(f"\nSample row:\n{feat_matrix.iloc[0]}")
    print("\n✅ features.py funcționează corect!")