"""
Clip classified hotspots to the actual India political boundary.
FIRMS 'India' query uses a bounding box, so it leaks in points from
Sri Lanka, Pakistan, Bhutan, etc. near the borders. This removes them.

Run from inside firms_fire_project/ with venv activated:
    python3 scripts/clip_to_india.py

Reads:  data/processed/firms_classified.csv
Writes: data/processed/firms_classified_india_only.csv
"""
import geopandas as gpd
import pandas as pd

INPUT_PATH = "data/processed/firms_classified.csv"
OUTPUT_PATH = "data/processed/firms_classified_india_only.csv"

# Reliable Natural Earth admin-0 countries boundary (widely used, well-maintained mirror)
NATURAL_EARTH_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} classified hotspots")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    # Drop any leftover index columns from earlier sjoin steps to avoid name clashes
    gdf = gdf.loc[:, ~gdf.columns.str.startswith(("index_right", "index_left"))]

    print("Fetching India boundary...")
    world = gpd.read_file(NATURAL_EARTH_URL)
    india = world[world["ADMIN"] == "India"].copy()
    if len(india) == 0:
        # column name fallback across Natural Earth versions
        india = world[world["NAME"] == "India"].copy()
    india = india.to_crs("EPSG:4326")
    # Small buffer (~5km) to avoid dropping legitimate coastal/border points
    # due to boundary simplification at this resolution
    india["geometry"] = india.buffer(0.05)

    clipped = gpd.sjoin(gdf, india[["geometry"]], how="inner", predicate="within")
    clipped = clipped.loc[:, ~clipped.columns.str.startswith("index_")]

    removed = len(df) - len(clipped)
    print(f"Removed {removed} hotspots outside India boundary ({removed/len(df)*100:.1f}%)")
    print(f"Remaining: {len(clipped)}")

    # Diagnostic: show lat/lon range of removed points to sanity-check the boundary
    removed_ids = set(df.index) - set(clipped.index)
    removed_df = df.loc[list(removed_ids)]
    print("\n=== Diagnostic: removed points lat/lon range ===")
    print(removed_df[["latitude", "longitude"]].describe())
    removed_df[["latitude", "longitude"]].to_csv("data/processed/_removed_points_debug.csv", index=False)
    print("Saved removed points -> data/processed/_removed_points_debug.csv (inspect manually)")

    print("\n=== Classification Summary (India only) ===")
    print(clipped["source_type"].value_counts())

    clipped = clipped.drop(columns="geometry")
    clipped.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")
