"""
For each FIRMS hotspot (already enriched with land-cover), find the nearest
OSM industrial/mining facility and compute the distance to it.

Run from inside firms_fire_project/ with venv activated:
    python3 scripts/nearest_facility.py

Reads:  data/processed/firms_with_landcover.csv
        data/raw/osm_industrial_facilities.geojson
Writes: data/processed/firms_with_landcover_and_distance.csv
"""
import geopandas as gpd
import pandas as pd

HOTSPOTS_PATH = "data/processed/firms_with_landcover.csv"
FACILITIES_PATH = "data/raw/osm_industrial_facilities.geojson"
OUTPUT_PATH = "data/processed/firms_with_landcover_and_distance.csv"

# Metric CRS for India — needed so distances come out in meters, not degrees
METRIC_CRS = "EPSG:7755"   # India-centric equal-area/projected CRS (meters)
# If EPSG:7755 isn't available in your pyproj install, EPSG:24378 or
# a UTM zone (e.g. EPSG:32644) also work reasonably for a country-wide estimate.

def load_hotspots():
    df = pd.read_csv(HOTSPOTS_PATH)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    return gdf

def load_facilities():
    gdf = gpd.read_file(FACILITIES_PATH)
    return gdf

def nearest_facility_join(hotspots_gdf, facilities_gdf):
    # Reproject both to a metric CRS so distance is in meters
    hs_m = hotspots_gdf.to_crs(METRIC_CRS)
    fac_m = facilities_gdf.to_crs(METRIC_CRS)

    joined = gpd.sjoin_nearest(
        hs_m,
        fac_m[["osm_id", "category", "name", "geometry"]],
        how="left",
        distance_col="distance_to_facility_m"
    )

    # sjoin_nearest can produce duplicate rows if multiple facilities are
    # exactly equidistant — keep only the first match per hotspot
    joined = joined[~joined.index.duplicated(keep="first")]

    joined = joined.rename(columns={
        "category": "nearest_facility_category",
        "name": "nearest_facility_name",
        "osm_id": "nearest_facility_osm_id",
    })

    # Bring back to the original (unprojected) geometry for saving as CSV
    joined = joined.drop(columns="geometry")
    return joined

if __name__ == "__main__":
    hotspots = load_hotspots()
    print(f"Loaded {len(hotspots)} hotspots")

    facilities = load_facilities()
    print(f"Loaded {len(facilities)} facilities")

    result = nearest_facility_join(hotspots, facilities)

    print(result[["latitude", "longitude", "landcover_class",
                   "nearest_facility_category", "distance_to_facility_m"]].head(10))

    print(f"\nDistance stats (meters):")
    print(result["distance_to_facility_m"].describe())

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")