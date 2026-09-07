import pandas as pd
from pathlib import Path
CSV = Path("data/processed/firms_with_landcover_and_distance.csv")
OUT = Path("data/processed")
if not CSV.exists():
    raise SystemExit(f"✗ File nahi mila: {CSV}. Pehle landcover_gee.py aur nearest_facility.py run karo.")
df = pd.read_csv(CSV)
print(f"Total hotspots: {len(df)}")
# ============================================================
# HEURISTIC 1: Location Persistence (flare detection)
# ============================================================
df["lat_round"] = df["latitude"].round(2)
df["lon_round"] = df["longitude"].round(2)
location_counts = (
    df.groupby(["lat_round", "lon_round"])
    .agg(
        detection_count=("latitude", "size"),
        unique_days=("acq_date", "nunique"),
        avg_bright=("bright_ti4", "mean"),
        avg_frp=("frp", "mean"),
    )
    .reset_index()
)
print(f"Unique locations (rounded to ~1km): {len(location_counts)}")
PERSISTENCE_DAYS_THRESHOLD = 2
BRIGHTNESS_THRESHOLD = 340
location_counts["likely_flare"] = (
    (location_counts["unique_days"] >= PERSISTENCE_DAYS_THRESHOLD)
    & (location_counts["avg_bright"] >= BRIGHTNESS_THRESHOLD)
)
df = df.merge(
    location_counts[["lat_round", "lon_round", "detection_count", "unique_days", "likely_flare"]],
    on=["lat_round", "lon_round"],
    how="left",
)
# ============================================================
# HEURISTIC 2: Facility proximity (industrial vs mining)
# ============================================================
FACILITY_DISTANCE_THRESHOLD_M = 1000
df["near_facility"] = df["distance_to_facility_m"] <= FACILITY_DISTANCE_THRESHOLD_M
# ============================================================
# HEURISTIC 3: Land-cover based natural vs agricultural fire
# ============================================================
FOREST_CLASSES = {"Tree cover", "Shrubland", "Grassland"}
CROP_CLASSES = {"Cropland"}
URBAN_CLASSES = {"Built-up"}
NON_FIRE_CLASSES = {"Permanent water bodies", "Herbaceous wetland", "Mangroves", "Snow and ice"}
# ============================================================
# FINAL 5-CLASS RULE
# ============================================================
def classify(row):
    if row["likely_flare"]:
        return "gas_flare"
    if row["near_facility"]:
        if row.get("nearest_facility_category") == "mining_quarry":
            return "mining_activity"
        return "industrial_fire"
    landcover = row.get("landcover_class")
    if landcover in FOREST_CLASSES:
        return "wildfire"
    if landcover in CROP_CLASSES:
        return "crop_burning"
    if landcover in URBAN_CLASSES:
        return "industrial_fire"
    if landcover in NON_FIRE_CLASSES:
        return "likely_false_positive"
    return "unclassified_other"

df["source_type"] = df.apply(classify, axis=1)
# ============================================================
# Save results
# ============================================================
out_path = OUT / "firms_classified.csv"
df.to_csv(out_path, index=False)
print(f"\n✓ Saved: {out_path}")
print("\n=== Classification Summary (5-class) ===")
print(df["source_type"].value_counts())
flare_locations = location_counts[location_counts["likely_flare"]]
print(f"\n=== Likely Gas Flare Sites ({len(flare_locations)} unique locations) ===")
if len(flare_locations) > 0:
    print(
        flare_locations[["lat_round", "lon_round", "detection_count", "unique_days", "avg_bright"]]
        .sort_values("detection_count", ascending=False)
        .to_string(index=False)
    )
else:
    print("Koi persistent high-brightness location nahi mila is data window mein.")
