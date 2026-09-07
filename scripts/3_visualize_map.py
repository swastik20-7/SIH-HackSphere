import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

GEOJSON = Path("data/processed/firms_hotspots_clean.geojson")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

if not GEOJSON.exists():
    raise SystemExit(f"✗ File nahi mila: {GEOJSON}. Pehle 2_clean_data.py run karo.")

gdf = gpd.read_file(GEOJSON)
print(f"Loaded {len(gdf)} hotspots")

fig, ax = plt.subplots(figsize=(10, 10))

# Agar confidence text hai (low/nominal/high) to category color use karo
# Confidence text hai (n/l/h) — hamesha categorical treat karo
gdf["confidence"] = gdf["confidence"].astype(str)

if gdf["confidence"].str.isnumeric().all():
    gdf.plot(
        ax=ax,
        column="confidence",
        cmap="Reds",
        markersize=4,
        alpha=0.6,
        legend=True,
    )
else:
    gdf.plot(
        ax=ax,
        column="confidence",
        categorical=True,
        legend=True,
        markersize=4,
        alpha=0.6,
        cmap="Reds",
    )

ax.set_title("VIIRS Thermal Hotspots — India", fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()

out_path = OUT / "firms_hotspots_map.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✓ Map saved: {out_path}")
plt.show()
