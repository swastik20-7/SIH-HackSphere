import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

RAW = Path("data/raw/firms_viirs_india_2months_raw.csv")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    raise SystemExit(f"✗ Raw file nahi mila: {RAW}. Pehle 1_download_data.py run karo.")

df = pd.read_csv(RAW)
print("Columns:", df.columns.tolist())
print(f"Total raw records: {len(df)}")

# VIIRS mein 'confidence' column text hoti hai: low / nominal / high
# MODIS mein numeric hoti hai (0-100). Dono cases handle kar rahe hain.
# Confidence numeric hai ya text, dono handle karo
df["confidence"] = df["confidence"].astype(str)

if df["confidence"].str.isnumeric().all():
    df_clean = df[df["confidence"].astype(int) >= 60].copy()
else:
    # VIIRS: 'n' = nominal, 'h' = high, 'l' = low
    df_clean = df[df["confidence"].isin(["n", "h", "nominal", "high"])].copy()

print(f"After confidence filter: {len(df_clean)}")

# GeoDataFrame banao
geometry = [Point(xy) for xy in zip(df_clean["longitude"], df_clean["latitude"])]
gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")

# Save
geojson_path = PROCESSED / "firms_hotspots_clean.geojson"
csv_path = PROCESSED / "firms_processed.csv"

gdf.to_file(geojson_path, driver="GeoJSON")
df_clean.to_csv(csv_path, index=False)

print(f"✓ Saved: {geojson_path}")
print(f"✓ Saved: {csv_path}")

print("\n=== Quick Stats ===")
print(f"Date range: {df_clean['acq_date'].min()} to {df_clean['acq_date'].max()}")
if "bright_ti4" in df_clean.columns:
    print(f"Avg brightness (ti4): {df_clean['bright_ti4'].mean():.2f} K")
