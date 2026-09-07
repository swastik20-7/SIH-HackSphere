"""
Train an XGBoost multiclass model on the heuristically-labeled hotspots
(source_type column) so future/new hotspots can be classified automatically
without re-running all the rule-based logic.

Run from inside firms_fire_project/ with venv activated:
    python3 scripts/train_model.py

Reads:  data/processed/firms_classified.csv
Writes: outputs/model.pkl (or models/model.pkl if you have that folder)
        outputs/feature_importance.png
        prints classification report + confusion matrix
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/firms_classified.csv"
MODEL_OUT_DIR = Path("outputs")
MODEL_OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Feature columns — everything the model is allowed to use
# ============================================================
FEATURE_COLS = [
    "bright_ti4",      # brightness temperature (channel 4)
    "bright_ti5",      # brightness temperature (channel 5)
    "frp",             # fire radiative power
    "confidence",      # FIRMS confidence (categorical: low/nominal/high)
    "daynight",        # D or N
    "detection_count", # how many times this location was detected
    "unique_days",     # persistence — over how many distinct days
    "distance_to_facility_m",  # distance to nearest industrial facility
    "landcover_class", # categorical land-cover type
]

TARGET_COL = "source_type"

def load_and_prepare(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")

    # Drop rows with missing target or missing critical features
    df = df.dropna(subset=[TARGET_COL])

    # Keep only columns we need (some may be missing depending on pipeline
    # state — filter to what's actually present)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"Warning: missing expected columns, skipping: {missing}")

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        X[col] = X[col].fillna("unknown")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Fill remaining numeric NaNs with median (simple, safe default)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    return X, y, available_features, encoders

if __name__ == "__main__":
    X, y, feature_cols, encoders = load_and_prepare(DATA_PATH)
    print(f"Using features: {feature_cols}")
    print(f"\nClass distribution:\n{y.value_counts()}")

    # Encode target labels
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Region-safe-ish split: random split with stratification on target
    # (a true spatial split by region is better for production; for the
    # hackathon prototype, a stratified random split is a reasonable start)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(target_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test, y_pred, target_names=target_encoder.classes_
    ))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
    print(cm_df)

    # Feature importance plot
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance — Thermal Hotspot Classification")
    plt.tight_layout()
    plt.savefig(MODEL_OUT_DIR / "feature_importance.png", dpi=150)
    print(f"\nSaved feature importance plot -> {MODEL_OUT_DIR / 'feature_importance.png'}")

    # Save model + encoders together (needed for inference later)
    joblib.dump({
        "model": model,
        "target_encoder": target_encoder,
        "feature_encoders": encoders,
        "feature_cols": feature_cols,
    }, MODEL_OUT_DIR / "model.pkl")
    print(f"Saved model -> {MODEL_OUT_DIR / 'model.pkl'}")