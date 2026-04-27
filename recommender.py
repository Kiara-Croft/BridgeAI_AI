"""
PASUL 4: MOTORUL DE RECOMANDARE
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from features import load_table, build_feature_matrix, FEATURE_SCHEMAS
from translations import translate

# Romanian product name cleaner 
_RO_NAME_MAP = [
    ("Căști",     "Headphones"),
    ("Casti",     "Headphones"),
    ("căști",     "headphones"),
    ("casti",     "headphones"),
    ("Tabletă",   "Tablet"),
    ("Tableta",   "Tablet"),
    ("tabletă",   "tablet"),
    ("tableta",   "tablet"),
    ("Televizor", "TV"),
    ("televizor", "TV"),
]

def _clean_name(name: str) -> str:
    if not name:
        return name
    s = str(name)
    for ro, en in _RO_NAME_MAP:
        s = en.join(s.split(ro)) if ro in s else s
    return s
 
from data_generator import (
    LAPTOP_PROFILES, MONITOR_PROFILES, SMARTPHONE_PROFILES, 
    HEADPHONE_PROFILES, TV_PROFILES, TABLET_PROFILES
)
 
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "BridgeAI_Dataset", "Data Cleaner", "Clean.db")

CATEGORY_MAP = {
    "laptop": ("laptops", LAPTOP_PROFILES),
    "laptops": ("laptops", LAPTOP_PROFILES),
    "monitor": ("monitors", MONITOR_PROFILES),
    "monitors": ("monitors", MONITOR_PROFILES),
    "smartphone": ("smartphones", SMARTPHONE_PROFILES),
    "telefon": ("smartphones", SMARTPHONE_PROFILES),
    "phone": ("smartphones", SMARTPHONE_PROFILES),
    "casti": ("headphones", HEADPHONE_PROFILES),
    "headphone": ("headphones", HEADPHONE_PROFILES),
    "casti gaming": ("gaming_headphones", HEADPHONE_PROFILES),
    "gaming headphone": ("gaming_headphones", HEADPHONE_PROFILES),
    "tv": ("tvs", TV_PROFILES),
    "televizor": ("tvs", TV_PROFILES),
    "tableta": ("tablets", TABLET_PROFILES),
    "tablet": ("tablets", TABLET_PROFILES),
}

INTENT_TO_PROFILE = {
    "gaming": "gamer_pro",
    "gaming entry": "gamer_entry",
    "gaming entry-level": "gamer_entry",
    "gaming budget": "gamer_entry",
    "gamer": "gamer_pro",
    "jocuri": "gamer_pro",
    "cs": "gamer_pro",
    "fps": "gamer_pro",
    "competitive gaming": "gamer_pro",
    
    "business": "business_professional",
    "munca": "business_professional",
    "lucru": "business_professional",
    "profesional": "business_professional",
    "birou": "business_professional",
    "office": "business_professional",
    
    "design": "designer_creator",
    "grafica": "designer_creator",
    "creator": "designer_creator",
    "video": "content_creator",
    "editing": "content_creator",
    "video editing": "content_creator",
    "photoshop": "designer_creator",
    "illustrator": "designer_creator",
    "premiere": "content_creator",
    "content creator": "content_creator",
    "content creation": "content_creator",
    "streaming": "content_creator",
    "youtube": "content_creator",
    "twitch": "content_creator",
    
    "programare": "student_performance",
    "programator": "student_performance",
    "coding": "student_performance",
    "development": "student_performance",
    "developer": "student_performance",
    
    "student": "student_budget",
    "student budget": "student_budget",
    "scoala": "student_budget",
    "facultate": "student_performance",
    "universitate": "student_performance",
    "student premium": "student_premium",
    "student apple": "student_premium",
    "student performant": "student_performance",
    "student performance": "student_performance",
    "inginerie": "student_performance",
    "cad": "student_performance",
    "autocad": "student_performance",
    "solidworks": "student_performance",
    
    "casual": "casual_user",
    "casual user": "casual_user",
    "navigare": "casual_user",
    "filme": "casual_user",
    "browsing": "casual_user",
    "netflix": "casual_user",
    "multimedia": "balanced_performance",
    "mixt": "balanced_performance",
    "mix": "balanced_performance",
    "uz general": "balanced_performance",
    "general": "balanced_performance",
    "balanced": "balanced_performance",
    
    "portabil": "portable_lightweight",
    "usor": "portable_lightweight",
    "lightweight": "portable_lightweight",
    "mobilitate": "portable_lightweight",
    "calatorii": "portable_lightweight",
    "transport": "portable_lightweight",
    
    "apple": "student_premium",
    "macbook": "student_premium",
    "mac": "student_premium",
    "premium": "student_premium",
    
    "gaming monitor": "gamer_competitive",
    "competitive": "gamer_competitive",
    "high refresh": "gamer_competitive",
    "144hz": "gamer_competitive",
    "240hz": "gamer_competitive",
    
    "oled": "creative_oled",
    "creativ": "creative_oled",
    "monitor grafica": "creative_oled",
    
    "monitor office": "office_standard",
    "monitor birou": "office_standard",
    "work monitor": "office_standard",
    
    "fotografie": "photo_video_editing",
    "foto": "photo_video_editing",
    "editing foto": "photo_video_editing",
    "editing video": "photo_video_editing",
    
    "monitor ieftin": "budget_multimedia",
    "monitor buget": "budget_multimedia",
    
    "monitor universal": "balanced_allround",
    "monitor general": "balanced_allround",
    
    "flagship": "flagship_seeker",
    "top phone": "flagship_seeker",
    "premium phone": "flagship_seeker",
    
    "poze": "photography_focus",
    "camera": "photography_focus",
    "phone camera": "photography_focus",
    "fotografii": "photography_focus",
    
    "social media": "social_media_heavy",
    "social": "social_media_heavy",
    "instagram": "social_media_heavy",
    "tiktok": "social_media_heavy",
    "facebook": "social_media_heavy",
    
    "buget mic": "essential_budget",
    "ieftin": "essential_budget",
    "budget": "essential_budget",
    "phone ieftin": "essential_budget",
    
    "power user": "power_user",
    "multitasking": "power_user",
    "productivitate": "power_user",
    
    "gaming mobile": "gaming_mobile",
    "mobile gaming": "gaming_mobile",
    "phone gaming": "gaming_mobile",
    
    "telefon echilibrat": "balanced_midrange",
    "midrange": "balanced_midrange",
    "mid-range": "balanced_midrange",
    
    "muzica": "audiophile_premium",
    "audiophile": "audiophile_premium",
    "music": "audiophile_premium",
    "podcast": "audiophile_premium",
    "ascultare": "audiophile_premium",
    
    "sport": "workout_wireless",
    "antrenament": "workout_wireless",
    "workout": "workout_wireless",
    "sala": "workout_wireless",
    "alergare": "workout_wireless",
    
    "anc": "commute_anc",
    "noise canceling": "commute_anc",
    "zgomot": "commute_anc",
    "metrou": "commute_anc",
    "naveta": "commute_anc",
    
    "casti ieftine": "budget_wireless",
    "casti buget": "budget_wireless",
    
    "voip": "calls_voip",
    "call": "calls_voip",
    "apel": "calls_voip",
    "conferinte": "calls_voip",
    "zoom": "calls_voip",
    "teams": "calls_voip",
    
    "gaming headset": "gaming_headset",
    "casti gaming": "gaming_headset",
    "jocuri casti": "gaming_headset",
    
    "cinema": "home_cinema",
    "home cinema": "home_cinema",
    "filme casa": "home_cinema",
    
    "smart tv": "smart_streaming",
    "netflix tv": "smart_streaming",
    "streaming tv": "smart_streaming",
    
    "tv ieftin": "budget_tv",
    "tv buget": "budget_tv",
    
    "tv gaming": "gaming_tv",
    "gaming tv": "gaming_tv",
    "console": "gaming_tv",
    "ps5": "gaming_tv",
    "xbox": "gaming_tv",
    
    "tableta productivitate": "productivity_tablet",
    "ipad work": "productivity_tablet",
    "tableta munca": "productivity_tablet",
    
    "tableta multimedia": "media_consumption",
    "tableta filme": "media_consumption",
    "tableta continut": "media_consumption",
    
    "tableta ieftina": "budget_tablet",
    "tableta buget": "budget_tablet",
}

BRAND_KEYWORDS = {
    "apple":     ["apple", "macbook", "mac book", "imac", "iphone", "ipad", "airpods"],
    "samsung":   ["samsung", "galaxy"],
    "asus":      ["asus", "rog", "zenbook", "vivobook", "tuf"],
    "dell":      ["dell", "xps", "inspiron", "alienware"],
    "lenovo":    ["lenovo", "thinkpad", "ideapad", "legion", "yoga"],
    "hp":        ["hp", "hewlett", "pavilion", "spectre", "envy", "omen"],
    "acer":      ["acer", "predator", "aspire", "nitro", "swift"],
    "msi":       ["msi", "stealth", "raider"],
    "lg":        ["lg"],
    "sony":      ["sony", "xperia"],
    "microsoft": ["microsoft", "surface"],
    "razer":     ["razer", "blade"],
    "huawei":    ["huawei", "matebook"],
    "xiaomi":    ["xiaomi", "redmi", "poco"],
    "oneplus":   ["oneplus", "one plus"],
    "google":    ["google", "pixel"],
    "realme":    ["realme"],
    "tcl":       ["tcl"],
    "motorola":  ["motorola", "moto"],
    "nokia":     ["nokia"],
}


def load_model(table_name):
    model_path = f"models/{table_name}_rf_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model lipsă: {model_path}. Rulează train_rf.py mai întâi!")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def find_best_profile(intent: str, profiles: list) -> dict:
    intent_lower = intent.lower()
    target_profile_name = INTENT_TO_PROFILE.get(intent_lower)
    
    if target_profile_name:
        for p in profiles:
            if p["name"] == target_profile_name:
                return p
 
    return profiles[0] if profiles else {}


def filter_by_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    if not brand:
        return df
    
    brand_lower = brand.lower()
    keywords = BRAND_KEYWORDS.get(brand_lower, [brand_lower])
    
    mask = pd.Series([False] * len(df), index=df.index)
    
    if "product_name" in df.columns:
        for kw in keywords:
            mask = mask | df["product_name"].str.lower().str.contains(kw, na=False)
    
    if "brand" in df.columns:
        for kw in keywords:
            mask = mask | df["brand"].str.lower().str.contains(kw, na=False)
    
    filtered = df[mask].copy()
    
    if len(filtered) == 0:
        print(f"⚠️  No {brand.upper()} products found. Showing all brands.")
        return df
    
    print(f"🏷️  Filtered by brand {brand.upper()}: {len(filtered)} products found")
    return filtered

def score_products(user_request: dict, top_n: int = 5) -> list:
    categorie = user_request.get("categorie", "").lower()
    scop = user_request.get("scop", "").lower()
    buget = float(user_request.get("buget", 999999))
    brand = user_request.get("brand", "").lower().strip()
 
    if categorie not in CATEGORY_MAP:
        return [{"error": f"Unknown category: {categorie}"}]
    
    table_name, profiles = CATEGORY_MAP[categorie]
 
    df_real = load_table(table_name)
    
    brand_filtered = False
    if brand:
        df_brand = filter_by_brand(df_real, brand)
        if len(df_brand) < len(df_real):
            brand_filtered = True
            df_real = df_brand
    
    if buget < 999999 and "price" in df_real.columns:
        df_real["price"] = pd.to_numeric(df_real["price"], errors="coerce")
        df_with_price = df_real[df_real["price"].notna()].copy()
        df_strict = df_with_price[df_with_price["price"] <= buget].copy()
        if len(df_strict) < 3:
            df_relaxed = df_with_price[df_with_price["price"] <= buget * 1.5].copy()
            if len(df_relaxed) >= 3:
                print(f"⚠️  Budget too strict, expanding to ±50%...")
                df_real = df_relaxed
            else:
                print(f"⚠️  Few products in budget. Showing best available.")
                df_real = df_relaxed if len(df_relaxed) > 0 else df_with_price
        else:
            df_real = df_strict

    # Storage filter if specified
    storage_req = float(user_request.get("storage", 0))
    if storage_req > 0 and "memorieinternaflash" in df_real.columns:
        df_real["memorieinternaflash"] = pd.to_numeric(df_real["memorieinternaflash"], errors="coerce")
        df_storage = df_real[df_real["memorieinternaflash"] >= storage_req].copy()
        if len(df_storage) >= 3:
            df_real = df_storage
            print(f"💾 Filtered by storage ≥{storage_req}GB: {len(df_real)} products")
    
    if len(df_real) == 0:
        return [{"error": "No products available in the database."}]
    
    df_real = df_real.reset_index(drop=True)
    
    print(f"📊 Produse evaluate: {len(df_real)}")
 
    feat_matrix = build_feature_matrix(df_real, table_name)
 
    if not profiles:
        scores = feat_matrix.mean(axis=1)
        df_real["_rf_score"] = scores
        df_real["_profile_used"] = "generic"
    else:
        profile = find_best_profile(scop, profiles)
 
        try:
            model_package = load_model(table_name)
            model = model_package["model"]
            model_feature_names = model_package["feature_names"]
        except FileNotFoundError as e:
            print(f"⚠️  {e}. Folosim scor simplu.")
            scores = feat_matrix.mean(axis=1)
            df_real["_rf_score"] = scores
            df_real["_profile_used"] = "fallback"
            model = None
        
        if model is not None:
            budget_tier = buget / 30000.0
            budget_tier = min(budget_tier, 5.0)
            
            rf_input = feat_matrix.copy()
            rf_input["profile_budget_tier"] = budget_tier
 
            for p in profiles:
                rf_input[f"profile_{p['name']}"] = 0.0
            
            selected_profile_col = f"profile_{profile['name']}"
            if selected_profile_col in rf_input.columns:
                rf_input[selected_profile_col] = 1.0
    
            for col in model_feature_names:
                if col not in rf_input.columns:
                    rf_input[col] = 0.0
            
            rf_input = rf_input[model_feature_names]
 
            scores = model.predict(rf_input)
            
            df_real["_rf_score"] = scores
            df_real["_profile_used"] = profile["name"]
 
    df_real = df_real.sort_values("_rf_score", ascending=False)
    
    results = []
    
    if len(df_real) <= top_n:
        selected_df = df_real.head(top_n)
    else:
        selected_indices = []
        
        if "price" in df_real.columns:
            df_sorted_price = df_real.copy()
            df_sorted_price = df_sorted_price[df_sorted_price["price"].notna()]
            
            if len(df_sorted_price) > 0:
                selected_indices.append(df_real.index[0])
                
                for idx in df_real.index[1:min(4, len(df_real))]:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                
                remaining_slots = top_n - len(selected_indices)
                if remaining_slots > 0 and len(df_real) > 10:
                    mid_start = min(10, len(df_real) // 3)
                    mid_end = min(20, len(df_real) // 2)
                    mid_range = df_real.iloc[mid_start:mid_end]
                    
                    for idx in mid_range.index[:remaining_slots]:
                        if idx not in selected_indices:
                            selected_indices.append(idx)
            else:
                selected_indices = df_real.head(top_n).index.tolist()
        else:
            selected_indices = df_real.head(top_n).index.tolist()
        
        selected_df = df_real.loc[selected_indices].sort_values("_rf_score", ascending=False)
    
    for i, (_, row) in enumerate(selected_df.iterrows()):
        score = row.get("_rf_score", 0)
        price = row.get("price", 0)
 
        product_info = {
            "rank": i + 1,
            "product_name": _clean_name(row.get("product_name", "N/A")),
            "price": f"{price:.2f} RON" if price else "N/A",
            "score": round(float(score), 4),
            "score_pct": f"{score*100:.1f}%",
            "url": row.get("url", ""),
            "profile_used": row.get("_profile_used", ""),
            "brand_filtered": brand_filtered,
        }
 
        if table_name == "laptops":
            product_info["specs"] = {
                "RAM": f"{row.get('capacity', 'N/A')} GB",
                "SSD": f"{row.get('capacitatessd', 'N/A')} GB",
                "CPU": str(row.get('frecventaturbomax', 'N/A')),
                "GPU": translate(row.get('placavideo', 'N/A')),
                "Display": f"{row.get('display_size', 'N/A')}\"",
                "Refresh": f"{row.get('ratarefresh', 'N/A')} Hz",
            }
        elif table_name == "smartphones":
            product_info["specs"] = {
                "RAM":     f"{row.get('ram', 'N/A')} GB",
                "Storage": f"{row.get('memorieinternaflash', 'N/A')} GB",
                "Battery": f"{row.get('capacitateacumulator', 'N/A')} mAh",
                "Display": f"{row.get('marimedisplay', 'N/A')}",
                "Refresh": f"{row.get('highrefreshrate', 'N/A')} Hz",
            }
        elif table_name == "monitors":
            product_info["specs"] = {
                "Size": f"{row.get('display_size', 'N/A')}\"",
                "Refresh": f"{row.get('refreshrate', 'N/A')} Hz",
                "Response": f"{row.get('timpderaspuns', 'N/A')} ms",
                "Panel": str(row.get('tippanel', 'N/A')),
            }
        elif table_name in ("headphones", "gaming_headphones"):
            product_info["specs"] = {
                "Type": translate(row.get('type', 'N/A')),
                "Mic": translate(row.get('microphone', 'N/A')),
                "ANC": translate(row.get('noise_canceling', 'N/A')),
                "Battery": f"{row.get('battery_life_music', 'N/A')} h",
            }
        elif table_name == "tvs":
            product_info["specs"] = {
                "Size": f"{row.get('diagonalainch', 'N/A')}\"",
                "Resolution": translate(row.get('imaginehd', 'N/A')),
                "Smart TV": translate(row.get('smarttv', 'N/A')),
                "Sound": f"{row.get('iesiresunetrms', 'N/A')} W",
            }
        elif table_name == "tablets":
            product_info["specs"] = {
                "Info": "Specs available in description"
            }
        
        results.append(product_info)
    
    return results


def format_results(results: list, user_request: dict) -> str:
    if not results or "error" in results[0]:
        return f"❌ {results[0].get('error', 'Eroare necunoscută')}"
    
    categorie = user_request.get("categorie", "produse")
    scop = user_request.get("scop", "")
    buget = user_request.get("buget", "nelimitat")
    brand = user_request.get("brand", "")
    
    output = []
    output.append(f"\n{'='*60}")
    brand_str = f" [{brand.upper()}]" if brand else ""
    output.append(f"🎯 Recomandări {categorie.upper()}{brand_str} pentru: {scop}")
    output.append(f"   Buget: {buget} RON | Profil: {results[0].get('profile_used', 'N/A')}")
    output.append(f"{'='*60}")
    
    for prod in results:
        output.append(f"\n#{prod['rank']} — {prod['product_name'][:55]}...")
        output.append(f"   💰 Preț:  {prod['price']}")
        output.append(f"   🎯 Scor:  {prod['score_pct']} potrivire")
        
        specs = prod.get("specs", {})
        if specs:
            spec_str = " | ".join([f"{k}: {v}" for k, v in specs.items() if str(v) != 'N/A' and str(v) != 'nan'])
            output.append(f"   🔧 Specs: {spec_str}")
        
        if prod.get("url"):
            output.append(f"   🔗 {prod['url'][:60]}...")
    
    output.append(f"\n{'='*60}")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("🧪 TEST RECOMMENDER\n")
    
    test_requests = [
        {"categorie": "laptop", "scop": "gaming", "buget": 5000, "brand": ""},
        {"categorie": "laptop", "scop": "programare", "buget": 5000, "brand": ""},
        {"categorie": "laptop", "scop": "casual", "buget": 7000, "brand": "apple"},
        {"categorie": "laptop", "scop": "student", "buget": 3000, "brand": ""},
        {"categorie": "monitor", "scop": "gaming", "buget": 3000, "brand": ""},
        {"categorie": "casti", "scop": "muzica", "buget": 800, "brand": "sony"},
    ]
    
    for req in test_requests:
        results = score_products(req, top_n=3)
        print(format_results(results, req))