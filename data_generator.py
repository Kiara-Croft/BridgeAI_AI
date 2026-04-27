"""
PASUL 2: GENERATORUL DE DATE SINTETICE

Acest script creează seturi de date de antrenament prin simularea comportamentului și preferințelor diferitelor profiluri de utilizatori. Acesta corelează specificațiile tehnice ale produselor cu necesitățile specifice (buget, performanță, brand) pentru a genera scoruri de relevanță necesare instruirii modelului de recomandare.

Funcții prezente:
- compute_match_score: Calculează un scor numeric (0-1) care reprezintă gradul de potrivire dintre un produs și un profil de utilizator, utilizând ponderi specifice pentru fiecare atribut tehnic.
- generate_training_data: Produce date de antrenament balansate pentru o categorie specifică, eșantionând produse care se potrivesc foarte bine sau foarte slab în două scenarii economice (buget constrâns și premium).
- generate_all: Automatizează procesul de generare pentru toate categoriile de produse (laptopuri, monitoare, telefoane etc.) și salvează rezultatele în fișiere CSV separate pentru antrenare și metadate.
"""

import numpy as np
import pandas as pd
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from features import load_table, build_feature_matrix, FEATURE_SCHEMAS

np.random.seed(42) 
random.seed(42) 

LAPTOP_PROFILES = [
    {
        "name": "gamer_entry",
        "description": "Gamer începător, echilibru preț-performanță",
        "weights": {
            "feat_price": 1.5,
            "feat_memoriededicata": 2.2,
            "feat_capacity": 1.5, 
            "feat_ratarefresh": 1.8,    
            "feat_category": 1.5, 
        },
        "budget_max": 6000,
        "category_prefer": "Gaming",
    },
    {
        "name": "gamer_pro",
        "description": "Gamer entuziast, performanță fără compromis",
        "weights": {
            "feat_price": 0.5,
            "feat_memoriededicata": 2.8, 
            "feat_capacity": 2.0,
            "feat_frecventaturbomax": 2.0,
            "feat_ratarefresh": 2.2,
            "feat_numarnuclee": 1.5,
        },
        "budget_max": 25000,
        "category_prefer": "Gaming",
    },
    {
        "name": "student_budget",
        "description": "Student axat pe economie și portabilitate",
        "weights": {
            "feat_price": 2.2,           
            "feat_capacity": 1.2,
            "feat_weight": 2.0,          
            "feat_capacitatessd": 1.0,
            "feat_sistemdeoperare": 1.0,
        },
        "budget_max": 4000,
        "category_prefer": "Multimedia",
    },
    {
        "name": "student_performance",
        "description": "Student cu nevoi de performanță (programare, inginerie, CAD)",
        "weights": {
            "feat_price": 1.3,
            "feat_capacity": 2.0,
            "feat_frecventaturbomax": 1.8,
            "feat_numarnuclee": 1.7,
            "feat_capacitatessd": 1.8,
            "feat_weight": 1.2,
        },
        "budget_max": 7000,
        "category_prefer": "Workstation",
    },
    {
        "name": "student_premium",
        "description": "Student/Profesionist care caută ecosistemul Apple și finisaje premium",
        "weights": {
            "feat_price": 0.8,           
            "feat_weight": 2.5,          
            "feat_sistemdeoperare": 2.8, 
            "feat_display_size": 1.5,    
            "feat_category": 1.5,
        },
        "budget_max": 12000,             
        "brand_prefer": "Apple",         
    },
    {
        "name": "casual_user",
        "description": "Utilizare casual: browsing, filme, office ușor",
        "weights": {
            "feat_price": 2.0,
            "feat_weight": 1.8,
            "feat_capacity": 1.0,
            "feat_display_size": 1.2,
            "feat_capacitatessd": 0.8,
        },
        "budget_max": 3500,
        "category_prefer": "Multimedia",
    },
    {
        "name": "balanced_performance",
        "description": "Echilibru între preț și performanță pentru uz mixt",
        "weights": {
            "feat_price": 1.5,
            "feat_capacity": 1.6,
            "feat_frecventaturbomax": 1.5,
            "feat_capacitatessd": 1.5,
            "feat_memoriededicata": 1.3,
            "feat_weight": 1.0,
        },
        "budget_max": 5500,
        "category_prefer": "Multimedia",
    },
    {
        "name": "business_professional",
        "description": "Mobilitate, securitate și aspect profesional",
        "weights": {
            "feat_price": 0.7,
            "feat_weight": 2.2,
            "feat_capacity": 1.8,
            "feat_sistemdeoperare": 2.5, 
            "feat_category": 2.0,        
        },
        "budget_max": 15000,
        "category_prefer": "Business",
    },
    {
        "name": "designer_creator",
        "description": "Putere de procesare grafică și acuratețe display",
        "weights": {
            "feat_price": 0.6,
            "feat_memoriededicata": 2.5,
            "feat_capacity": 2.0,
            "feat_capacitatessd": 1.8,
            "feat_category": 2.5,        
            "feat_display_size": 1.8,
        },
        "budget_max": 30000,
        "category_prefer": "Grafica",
    },
    {
        "name": "content_creator",
        "description": "Streaming, editing video, multitasking",
        "weights": {
            "feat_price": 1.0,
            "feat_capacity": 2.2,
            "feat_numarnuclee": 2.0,
            "feat_capacitatessd": 2.0,
            "feat_memoriededicata": 1.8,
            "feat_display_size": 1.5,
        },
        "budget_max": 8000,
        "category_prefer": "Multimedia",
    },
    {
        "name": "portable_lightweight",
        "description": "Prioritate pe mobilitate și autonomie",
        "weights": {
            "feat_price": 1.2,
            "feat_weight": 2.8,
            "feat_display_size": 0.5,
            "feat_capacity": 1.2,
            "feat_sistemdeoperare": 1.5,
        },
        "budget_max": 6000,
        "category_prefer": "Ultraportabil",
    }
]

MONITOR_PROFILES = [
    {
        "name": "gamer_competitive",
        "weights": {"feat_refreshrate": 3.0, "feat_timpderaspuns": 2.5, "feat_price": 0.8},
        "budget_max": 7000,
    },
    {
        "name": "creative_oled",
        "weights": {"feat_tippanel": 3.0, "feat_brightness": 2.0, "feat_display_size": 1.5, "feat_recomandatpentru": 2.0},
        "budget_max": 12000,
    },
    {
        "name": "office_standard",
        "weights": {"feat_price": 2.0, "feat_display_size": 1.8, "feat_recomandatpentru": 1.5},
        "budget_max": 2500,
    },
    {
        "name": "photo_video_editing",
        "weights": {"feat_tippanel": 2.8, "feat_brightness": 2.5, "feat_display_size": 2.0, "feat_price": 1.0},
        "budget_max": 8000,
    },
    {
        "name": "budget_multimedia",
        "weights": {"feat_price": 2.5, "feat_display_size": 1.5, "feat_brightness": 1.0},
        "budget_max": 1500,
    },
    {
        "name": "balanced_allround",
        "weights": {"feat_refreshrate": 1.5, "feat_display_size": 1.5, "feat_price": 1.5, "feat_brightness": 1.2},
        "budget_max": 3000,
    }
]

SMARTPHONE_PROFILES = [
    {
        "name": "flagship_seeker",
        "weights": {"feat_main_camera": 2.5, "feat_refresh_rate": 2.0, "feat_memory_ram": 1.5, "feat_price": 0.5},
        "budget_max": 9000,
    },
    {
        "name": "social_media_heavy",
        "weights": {"feat_main_camera": 2.0, "feat_internal_storage": 2.0, "feat_battery_capacity": 1.8},
        "budget_max": 5000,
    },
    {
        "name": "essential_budget",
        "weights": {"feat_price": 2.5, "feat_battery_capacity": 2.0},
        "budget_max": 1800,
    },
    {
        "name": "photography_focus",
        "weights": {"feat_main_camera": 3.0, "feat_internal_storage": 2.0, "feat_price": 1.0},
        "budget_max": 7000,
    },
    {
        "name": "power_user",
        "weights": {"feat_memory_ram": 2.5, "feat_internal_storage": 2.0, "feat_battery_capacity": 2.0, "feat_refresh_rate": 1.8},
        "budget_max": 6000,
    },
    {
        "name": "gaming_mobile",
        "weights": {"feat_refresh_rate": 2.8, "feat_memory_ram": 2.5, "feat_battery_capacity": 2.0, "feat_price": 1.0},
        "budget_max": 5500,
    },
    {
        "name": "balanced_midrange",
        "weights": {"feat_price": 1.8, "feat_battery_capacity": 1.8, "feat_main_camera": 1.5, "feat_memory_ram": 1.5},
        "budget_max": 3000,
    }
]

HEADPHONE_PROFILES = [
    {
        "name": "audiophile_premium",
        "weights": {"feat_noise_canceling": 2.5, "feat_type": 1.5, "feat_headphone_type": 2.0, "feat_price": 0.8},
        "budget_max": 4000,
    },
    {
        "name": "workout_wireless",
        "weights": {"feat_type": 3.0, "feat_battery_life_music": 2.0, "feat_price": 1.5},
        "budget_max": 1200,
    },
    {
        "name": "commute_anc",
        "weights": {"feat_noise_canceling": 3.0, "feat_type": 2.5, "feat_battery_life_music": 2.0, "feat_price": 1.2},
        "budget_max": 2000,
    },
    {
        "name": "budget_wireless",
        "weights": {"feat_price": 2.5, "feat_type": 2.0, "feat_battery_life_music": 1.5},
        "budget_max": 500,
    },
    {
        "name": "calls_voip",
        "weights": {"feat_microphone": 3.0, "feat_noise_canceling": 2.0, "feat_type": 2.0, "feat_price": 1.0},
        "budget_max": 1500,
    },
    {
        "name": "gaming_headset",
        "weights": {"feat_microphone": 2.8, "feat_headphone_type": 2.0, "feat_price": 1.2},
        "budget_max": 1800,
    }
]

TV_PROFILES = [
    {
        "name": "home_cinema",
        "weights": {"feat_diagonalainch": 2.5, "feat_imaginehd": 2.8, "feat_smarttv": 2.0, "feat_iesiresunetrms": 2.0, "feat_price": 1.0},
        "budget_max": 8000,
    },
    {
        "name": "smart_streaming",
        "weights": {"feat_smarttv": 3.0, "feat_imaginehd": 2.0, "feat_diagonalainch": 1.5, "feat_price": 1.5},
        "budget_max": 4000,
    },
    {
        "name": "budget_tv",
        "weights": {"feat_price": 2.5, "feat_diagonalainch": 1.5, "feat_smarttv": 1.0},
        "budget_max": 2000,
    },
    {
        "name": "gaming_tv",
        "weights": {"feat_imaginehd": 2.5, "feat_diagonalainch": 2.0, "feat_price": 1.2},
        "budget_max": 6000,
    },
]

TABLET_PROFILES = [
    {
        "name": "productivity_tablet",
        "weights": {"feat_price": 1.2},
        "budget_max": 5000,
    },
    {
        "name": "media_consumption",
        "weights": {"feat_price": 1.8},
        "budget_max": 3000,
    },
    {
        "name": "budget_tablet",
        "weights": {"feat_price": 2.5},
        "budget_max": 1500,
    },
]
PROFILES_MAP = {
    "laptops": LAPTOP_PROFILES,
    "monitors": MONITOR_PROFILES,
    "smartphones": SMARTPHONE_PROFILES,
    "headphones": HEADPHONE_PROFILES,
    "gaming_headphones": HEADPHONE_PROFILES,
    "tvs": TV_PROFILES,
    "tablets": TABLET_PROFILES,
}

def compute_match_score(product_features: pd.Series, profile: dict, brand_target: str = None) -> float:
    profile_weights = profile.get("weights", {}) 
    total_weight = 0.0 
    weighted_sum = 0.0 
    
    for feat_name, feat_value in product_features.items():
        weight = profile_weights.get(feat_name, 0.3) 
        weighted_sum += feat_value * weight 
        total_weight += weight 
    
    if total_weight == 0: return 0.0 

    raw_score = weighted_sum / total_weight 
    product_brand = str(product_features.get("_brand", "")).lower()
    if brand_target and brand_target.lower() in product_brand: raw_score += 0.2 
        
    return min(1.0, max(0.0, raw_score))

def generate_training_data(table_name, profiles, n_samples_per_profile=300):
    print(f"\n[GENERATOR] Procesare tabel: {table_name}")

    df = load_table(table_name)
    feat_matrix = build_feature_matrix(df, table_name)
    
    if 'brand' in df.columns:
        feat_matrix['_brand'] = df['brand'].str.lower()
    
    print(f"  → {len(df)} produse, {len(feat_matrix.columns) - 1} features active")
    
    training_rows = []
    
    for profile in profiles:
        profile_name = profile["name"]
        
        for budget_scenario in ["budget_constrained", "premium_flexible"]:
            
            temp_weights = profile["weights"].copy()
            
            if budget_scenario == "budget_constrained":
                temp_weights["feat_price"] = max(temp_weights.get("feat_price", 1.0), 2.5)
                current_budget_limit = profile.get("budget_max", 3000) * 0.6
            else:
                temp_weights["feat_price"] = 0.5 
                current_budget_limit = profile.get("budget_max", 3000) * 1.5

            temp_profile = profile.copy()
            temp_profile["weights"] = temp_weights

            brand_target = profile.get("brand_prefer", None)
            scores = feat_matrix.apply(
                lambda row: compute_match_score(row, temp_profile, brand_target=brand_target), 
                axis=1
            )
            
            n = min(n_samples_per_profile // 2, len(df))
            sorted_idx = scores.argsort()
            
            top_idx = sorted_idx.tail(n // 2).tolist()
            bot_idx = sorted_idx.head(n // 2).tolist()
            sampled_idx = top_idx + bot_idx
            
            for idx in sampled_idx:
                row = {k: v for k, v in feat_matrix.iloc[idx].to_dict().items() if not k.startswith("_")}
                
                row["profile_budget_tier"] = current_budget_limit / 30000.0
                
                for p in profiles:
                    row[f"profile_{p['name']}"] = 1.0 if p["name"] == profile_name else 0.0
                
                row["target_score"] = scores.iloc[idx]
                
                row["_product_name"] = df.iloc[idx]["product_name"] if "product_name" in df.columns else "N/A"
                row["_profile"] = f"{profile_name}_{budget_scenario}"
                row["_price"] = df.iloc[idx]["price"] if "price" in df.columns else 0
                
                training_rows.append(row)

    training_df = pd.DataFrame(training_rows)
    
    noise = np.random.normal(0, 0.02, len(training_df))
    training_df["target_score"] = (training_df["target_score"] + noise).clip(0, 1)
    
    print(f"  → Generat {len(training_df)} rânduri de antrenament.")
    return training_df

def generate_all():
    tables_to_process = [
        ("laptops", LAPTOP_PROFILES),
        ("monitors", MONITOR_PROFILES),
        ("smartphones", SMARTPHONE_PROFILES),
        ("headphones", HEADPHONE_PROFILES),
        ("gaming_headphones", HEADPHONE_PROFILES),
        ("tvs", TV_PROFILES),
        ("tablets", TABLET_PROFILES),
    ]
    
    os.makedirs("training_data", exist_ok=True)
    
    summary = {}
    
    for table_name, profiles in tables_to_process:
        training_df = generate_training_data(table_name, profiles, n_samples_per_profile=300)
        
        meta_cols = [c for c in training_df.columns if c.startswith("_")]
        feature_cols = [c for c in training_df.columns if not c.startswith("_")]
        
        output_path = f"training_data/{table_name}_training.csv"
        training_df[feature_cols].to_csv(output_path, index=False)
        
        meta_path = f"training_data/{table_name}_meta.csv"
        training_df[meta_cols + ["target_score"]].to_csv(meta_path, index=False)
        
        summary[table_name] = {
            "examples": len(training_df),
            "features": len(feature_cols) - 1,
            "output": output_path
        }
        
        print(f"  ✅ Salvat: {output_path}")
    
    print("\n" + "="*50)
    print("SUMAR GENERARE DATE:")
    for table, info in summary.items():
        print(f"  {table}: {info['examples']} exemple, {info['features']} features")
    print("="*50)
    
    return summary

if __name__ == "__main__":
    import os
    generate_all()
    print("\n✅ Date sintetice generate cu succes!")