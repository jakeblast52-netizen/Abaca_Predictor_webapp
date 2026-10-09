import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ── CONFIG ────────────────────────────────────────────────────────────────────
FILE_PATH   = r"C:\Users\wilmer hehe\OneDrive\RESEARCH\abaca_real_200.xlsx"
MODEL_OUT   = "abaca_rf_model.pkl"
TARGET      = "girth_cm"
# ─────────────────────────────────────────────────────────────────────────────


def load_and_clean(path):
    df = pd.read_excel(path)

    df["moisture_pct"]  = df["moisture"]  * 100  
    df["humidity_pct"]  = df["humidity"]  * 100   
    df["sun_shade_pct"] = df["sun_shade"] * 100   

    df = df.dropna(subset=[TARGET])
    df = df[(df["height_cm"] > 50) & (df[TARGET] > 0)]

    return df


FEATURES = [
    "height_cm",       # cm        range: 76–236
    "leaf_count",      #           range: 2–6
    "moisture_pct",    # %         range: 87–99
    "soil_pH",         #           range: 3.5–9.0
    "temperature",     # °C        range: 27.5–32.8
    "humidity_pct",    # %         range: 56–73
    "sun_shade_pct",   # %         range: 44–69
]


def train(df):
    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores   = cross_val_score(model, X, y, cv=kf, scoring="r2")
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
    mae_scores  = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

    # Final fit on full dataset
    model.fit(X, y)

    return model, r2_scores, rmse_scores, mae_scores


def predict(model, height_cm, leaf_count, moisture_pct, soil_pH,
            temperature, humidity_pct, sun_shade_pct):
    sample = pd.DataFrame([{
        "height_cm":      height_cm,
        "leaf_count":     leaf_count,
        "moisture_pct":   moisture_pct,
        "soil_pH":        soil_pH,
        "temperature":    temperature,
        "humidity_pct":   humidity_pct,
        "sun_shade_pct":  sun_shade_pct,
    }])
    return float(model.predict(sample)[0])


def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    df = load_and_clean(FILE_PATH)
    print(f"✅ Loaded {len(df)} samples\n")

    model, r2, rmse, mae = train(df)

    # ── PERFORMANCE ──────────────────────────────────────────────────────────
    print("=" * 52)
    print("MODEL PERFORMANCE  (5-fold cross-validation)")
    print("=" * 52)
    print(f"  R²   : {r2.mean():.4f}  ±  {r2.std():.4f}")
    print(f"  RMSE : {rmse.mean():.4f}  ±  {rmse.std():.4f} ")
    print(f"  MAE  : {mae.mean():.4f}  ±  {mae.std():.4f} ")

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("FEATURE IMPORTANCE")
    print("=" * 52)
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 400)
        print(f"  {feat:<18}: {imp*100:5.2f}%  {bar}")

    # ── VALID INPUT RANGES ───────────────────────────────────────────────────
    X = df[FEATURES]
    print("\n" + "=" * 52)
    print("VALID INPUT RANGES  (from your 200 samples)")
    print("=" * 52)
    for f in FEATURES:
        unit = "%" if "pct" in f else ("°C" if f == "temperature" else ("cm" if "cm" in f else ""))
        print(f"  {f:<18}: {X[f].min():.1f} – {X[f].max():.1f} {unit}")

    # ── SENSITIVITY: sun_shade_pct ────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("SENSITIVITY: sun_shade_pct  (other features at median)")
    print("=" * 52)
    med = X.median().to_dict()
    for shade in range(44, 70, 3):
        med["sun_shade_pct"] = shade
        p = model.predict(pd.DataFrame([med]))[0]
        print(f"  {shade}% shade → {p:.2f} cm")

    # ── SAMPLE PREDICTION ────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("SAMPLE PREDICTION")
    print("=" * 52)
    pred = predict(
        model,
        height_cm=178,
        leaf_count=4,
        moisture_pct=96,   
        soil_pH=5.67,
        temperature=29.94,
        humidity_pct=67,    
        sun_shade_pct=70,  
    )
    print(f"  Predicted girth: {pred:.2f} cm")

    # ── SAVE MODEL ───────────────────────────────────────────────────────────
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({
            "model":       model,
            "features":    FEATURES,
            "data_ranges": {feat: (float(X[feat].min()), float(X[feat].max()))
                            for feat in FEATURES},
            "note": (
                "moisture_pct, humidity_pct, sun_shade_pct are in PERCENT "
                "(e.g. 96, 67, 55) — not decimals. "
                "Predictions outside data ranges will extrapolate poorly."
            ),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()