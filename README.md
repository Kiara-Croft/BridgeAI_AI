# BridgeAI — Ghid complet pas cu pas

## Arhitectura sistemului

```
Utilizator: "laptop gaming 5000 lei"
        │
        ▼
┌─────────────────┐
│   NLP Bridge    │  ← Claude API extrage intenția structurată
│  (main.py)      │    { categorie, scop, buget }
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Engine │  ← Transformă produsele din DB în numere (0-1)
│  (features.py)  │    RAM 16GB → 0.8, Preț 5000 → 0.6, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Random Forest  │  ← Scorează FIECARE produs: "cât de potrivit e?"
│  (recommender)  │    Laptop A: 0.601 | Laptop B: 0.598 | ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prezentare     │  ← Claude API formulează răspuns natural
│  (main.py)      │    "Recomand X pentru că are GPU dedicat..."
└─────────────────┘
```

---

## Fișierele proiectului

| Fișier | Rol | Când rulezi |
|--------|-----|-------------|
| `features.py` | Definește și normalizează features | Automat (importat) |
| `data_generator.py` | Generează date sintetice de training | O singură dată |
| `train_rf.py` | Antrenează modelele Random Forest | O singură dată |
| `recommender.py` | Scorează produse cu RF antrenat | La fiecare cerere |
| `main.py` | NLP + interfața conversațională | Rulat de utilizator |

---

## Cum rulezi (de la zero)

```bash
# 1. Instalează dependențele
pip install scikit-learn pandas numpy

# 2. Generează datele de training (o singură dată)
python3 data_generator.py

# 3. Antrenează modelele RF (o singură dată, ~30 secunde)
python3 train_rf.py

# 4. Pornește chatbot-ul
python3 main.py

# OPȚIONAL: Pentru prezentare naturală (Claude API)
export ANTHROPIC_API_KEY="sk-ant-..."
python3 main.py
```

---

## Înțelegerea Random Forest — de ce funcționează

### Problema noastră
Avem 948 laptopuri și un utilizator care spune "vreau ceva bun pentru gaming".
Trebuie să ordonăm cele 948 laptopuri de la "cel mai potrivit" la "cel mai nepotrivit".

### Soluția cu RF

**Pasul 1 — Feature engineering** (`features.py`):
Transformăm fiecare laptop din "text în DB" în "vector de numere":
```
Laptop Acer Gaming → [0.60, 0.80, 1.00, 0.45, ...] ← 12 numere între 0-1
                       preț  RAM  GPU   greutate
```

**Pasul 2 — Date sintetice** (`data_generator.py`):
Inventăm scenarii de genul:
```
"Profil GAMER + Laptop cu GPU Dedicat + 16GB RAM + 165Hz → scor 0.85"
"Profil GAMER + Laptop fără GPU + 8GB RAM + 60Hz → scor 0.23"
```
Generăm 1800 astfel de exemple pentru laptopuri.

**Pasul 3 — Antrenare RF** (`train_rf.py`):
RF construiește 200 de arbori de decizie, fiecare pe un subset random din date.
La predicție, face MEDIA scorurilor din toți arborii.

```
Arbore 1: "Are GPU dedicat? DA → scor 0.7"
Arbore 2: "Refresh > 144Hz? DA → scor 0.8"  
Arbore 3: "RAM > 16GB? NU → scor 0.5"
...
MEDIE → 0.67 (scorul final pentru acel laptop)
```

**Pasul 4 — Inferență** (`recommender.py`):
La runtime, RF scorează FIECARE produs real din DB și returnează top N.

### De ce RF și nu alte modele?

| Criteriu | Random Forest | Neural Network | KNN |
|----------|--------------|----------------|-----|
| Date necesare | **Puține (OK cu 1000)** | Multe (>10k) | Mediu |
| Interpretabil | **DA** (feature importance) | Nu | Partial |
| Timp antrenare | **Rapid (secunde)** | Lent (minute-ore) | Instant |
| Overfitting risk | **Mic** (by design) | Mare | Mediu |
| Parametri de tunat | Puțini | Mulți | Puțini |

---

## Rezultatele antrenării

```
Categorie          R² Test    RMSE     OOB
laptops            0.9239    0.0401   0.9275  ✅ Excelent
monitors           0.9462    0.0341   0.9381  ✅ Excelent
smartphones        0.9498    0.0353   0.9562  ✅ Excelent
headphones         0.9307    0.0427   0.9395  ✅ Excelent
gaming_headphones  0.9615    0.0313   0.9575  ✅ Excelent
```

**R² = 0.92** înseamnă că RF explică 92% din variația scorurilor.
**RMSE = 0.04** înseamnă că greșește în medie cu 4 puncte procentuale (scara 0-1).

---

## Cum să îmbunătățești sistemul

### 1. Adaugă mai multe profiluri de utilizator
În `data_generator.py`, adaugă un nou profil în `LAPTOP_PROFILES`:
```python
{
    "name": "streamer",
    "description": "Creator de conținut, live streaming",
    "weights": {
        "feat_capacity": 2.0,       # RAM mult pentru OBS
        "feat_frecventaturbomax": 1.8,
        "feat_cameraweb": 2.0,      # Webcam important
        "feat_placavideo": 1.5,
    },
    "budget_max": 8000,
    "category_prefer": "Multimedia",
}
```
Apoi rulează din nou `data_generator.py` și `train_rf.py`.

### 2. Colectează feedback real
Când un utilizator alege un produs, salvezi:
```python
{"cerinta": intent, "produs_ales": product_id, "scor_rf": score}
```
Și poți re-antrena RF cu aceste date reale (mai bune decât sinteticele).

### 3. Ajustează hiperparametrii RF
În `train_rf.py`, experimentează cu:
```python
model = RandomForestRegressor(
    n_estimators=500,   # mai mulți arbori = mai stabil
    max_depth=12,       # mai adânc = mai complex (risc overfitting)
    min_samples_leaf=1, # permite frunze mai specifice
)
```

### 4. Adaugă tablete și TVs
Definește profiluri în `data_generator.py` pentru `tvs` și `tablets`
(momentan folosesc scor generic).

---

## Concepte cheie de reținut

**Overfitting**: Modelul memorează datele de training dar nu generalizează.
Semn: R² train >> R² test. Soluție: mărești `min_samples_leaf`, reduci `max_depth`.

**Feature importance**: RF îți spune ce features contează cel mai mult.
Ex: la laptopuri, `capacitatessd` și `budget_tier` sunt cele mai importante.

**Out-of-Bag (OOB) score**: Evaluare "gratuită" — fiecare arbore e testat
pe datele pe care nu le-a văzut la antrenare. Echivalent cu cross-validation.

**Date sintetice vs. date reale**: Datele sintetice sunt suficiente pentru start,
dar datele reale de la utilizatori reali îmbunătățesc semnificativ acuratețea.