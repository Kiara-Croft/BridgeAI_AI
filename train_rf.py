"""
PASUL 3: ANTRENAREA RANDOM FOREST
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

def train_model_for_category(table_name, verbose=True):
    training_path = f"training_data/{table_name}_training.csv"
    
    if not os.path.exists(training_path):
        print(f"❌ Fișier training lipsă: {training_path}")
        print("   Rulează mai întâi: python3 data_generator.py")
        return None
    
    if verbose:
        print(f"\n{'='*55}")
        print(f"  ANTRENARE RF PENTRU: {table_name.upper()}")
        print(f"{'='*55}")

    df = pd.read_csv(training_path)

    target_col = "target_score"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if verbose:
        print(f"\n📊 Date de training:")
        print(f"   Exemple: {len(df)}")
        print(f"   Features: {len(X.columns)}")
        print(f"   Target range: {y.min():.3f} → {y.max():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if verbose:
        print(f"\n📋 Split train/test:")
        print(f"   Train: {len(X_train)} exemple")
        print(f"   Test:  {len(X_test)} exemple")

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        oob_score=True,   
    )
    
    if verbose:
        print(f"\n🌲 Antrenare Random Forest...")
    
    model.fit(X_train, y_train)
  
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    oob = model.oob_score_
    
    if verbose:
        print(f"\n📈 Rezultate evaluare:")
        print(f"   R² Train:  {r2_train:.4f}  (RMSE: {rmse_train:.4f})")
        print(f"   R² Test:   {r2_test:.4f}  (RMSE: {rmse_test:.4f})")
        print(f"   OOB Score: {oob:.4f}")
        
        if r2_test > 0.85:
            print(f"   ✅ Model excelent!")
        elif r2_test > 0.70:
            print(f"   ✅ Model bun")
        elif r2_test > 0.50:
            print(f"   ⚠️  Model acceptabil (mai adaugă date sau ajustează features)")
        else:
            print(f"   ❌ Model slab (revizuiește features și datele de training)")

        gap = r2_train - r2_test
        if gap > 0.15:
            print(f"   ⚠️  Posibil overfitting (gap train-test: {gap:.3f})")
        else:
            print(f"   ✅ Fără overfitting semnificativ (gap: {gap:.3f})")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    if verbose:
        print(f"\n🔍 Top 5 features importante:")
        for _, row in feature_importance.head(5).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"   {row['feature']:<30} {bar} {row['importance']:.4f}")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{table_name}_rf_model.pkl"
    
    model_package = {
        "model": model,
        "feature_names": list(X.columns),
        "feature_importance": feature_importance,
        "metrics": {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "oob_score": oob,
        },
        "table_name": table_name,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    if verbose:
        print(f"\n💾 Model salvat: {model_path}")
    
    return model_package

def train_all_models():
    categories = [
        "laptops",
        "monitors", 
        "smartphones",
        "headphones",
        "gaming_headphones",
        "tvs",
        "tablets",
    ]
    
    results = {}
    
    for category in categories:
        try:
            result = train_model_for_category(category)
            if result:
                results[category] = result["metrics"]
        except Exception as e:
            print(f"\n❌ Eroare la {category}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*55}")
    print("SUMAR ANTRENARE:")
    print(f"{'='*55}")
    print(f"{'Categorie':<25} {'R² Test':>8} {'RMSE':>8} {'OOB':>8}")
    print("-" * 55)
    for cat, metrics in results.items():
        print(f"{cat:<25} {metrics['r2_test']:>8.4f} {metrics['rmse_test']:>8.4f} {metrics['oob_score']:>8.4f}")
    
    return results

if __name__ == "__main__":
    print("🌲 BridgeAI — Antrenare Random Forest")
    print("Asigură-te că ai rulat mai întâi data_generator.py\n")
    
    train_all_models()
    print("\n✅ Toate modelele au fost antrenate și salvate!")