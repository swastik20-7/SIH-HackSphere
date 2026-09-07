"""
Fetch industrial facility locations (refineries, power plants, steel plants,
mining areas) across India from OpenStreetMap via the Overpass API.

Run from inside firms_fire_project/ with venv activated:
    python3 scripts/fetch_osm_facilities.py

Writes: data/raw/osm_industrial_facilities.geojson
"""
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape
import time
import json

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
HEADERS = {
    "User-Agent": "SIH2026-ThermalHotspot-Research/1.0"
}

# One query per facility category — smaller queries are less likely to time out
QUERIES = {
    "industrial_landuse": """
        [out:json][timeout:180];
        area["ISO3166-1"="IN"]->.india;
        (
          way["landuse"="industrial"](area.india);
          relation["landuse"="industrial"](area.india);
        );
        out center;
    """,
    "power_plant": """
        [out:json][timeout:180];
        area["ISO3166-1"="IN"]->.india;
        (
          node["power"="plant"](area.india);
          way["power"="plant"](area.india);
        );
        out center;
    """,
    "refinery_works": """
        [out:json][timeout:180];
        area["ISO3166-1"="IN"]->.india;
        (
          node["man_made"="works"](area.india);
          way["man_made"="works"](area.india);
        );
        out center;
    """,
    "mining_quarry": """
        [out:json][timeout:180];
        area["ISO3166-1"="IN"]->.india;
        (
          way["landuse"="quarry"](area.india);
          node["man_made"="mineshaft"](area.india);
        );
        out center;
    """,
}

def fetch_category(name, query):
    print(f"Fetching {name} ...")
    for attempt in range(3):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=200)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            print(f"  {name}: {len(elements)} elements")
            return elements
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    print(f"  {name}: FAILED after 3 attempts")
    return []

def elements_to_points(elements, category):
    rows = []
    for el in elements:
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:  # way/relation -> use center
            center = el.get("center")
            if not center:
                continue
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        name = el.get("tags", {}).get("name", "")
        rows.append({
            "osm_id": el["id"],
            "category": category,
            "name": name,
            "latitude": lat,
            "longitude": lon,
        })
    return rows

if __name__ == "__main__":
    all_rows = []
    for cat_name, query in QUERIES.items():
        elements = fetch_category(cat_name, query)
        all_rows.extend(elements_to_points(elements, cat_name))
        time.sleep(2)  # be polite to the public Overpass server

    df = pd.DataFrame(all_rows)
    print(f"\nTotal facilities fetched: {len(df)}")
    print(df["category"].value_counts())

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )

    out_path = "data/raw/osm_industrial_facilities.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved -> {out_path}")