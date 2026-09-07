"""
Land-cover integration using Google Earth Engine (ESA WorldCover 2021, 10m).
Batched version — safe for large FIRMS datasets (thousands of rows).

Run from inside firms_fire_project/ with venv activated:
    python3 scripts/landcover_gee.py
"""
import ee
import pandas as pd
import time

PROJECT_ID = "sih-thermal-hotspot"
BATCH_SIZE = 500  # points per Earth Engine request

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

def get_landcover_batch(df_batch, lat_col, lon_col):
    """Sample land-cover for one batch of rows. df_batch index is used as row_id."""
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')

    features = []
    for idx, row in df_batch.iterrows():
        pt = ee.Geometry.Point([row[lon_col], row[lat_col]])
        features.append(ee.Feature(pt, {'row_id': int(idx)}))
    fc = ee.FeatureCollection(features)

    sampled = worldcover.sampleRegions(collection=fc, scale=10, geometries=False)
    results = sampled.getInfo()['features']
    return {f['properties']['row_id']: f['properties'].get('Map') for f in results}

def get_landcover_for_points(df, lat_col='latitude', lon_col='longitude', batch_size=BATCH_SIZE):
    df = df.reset_index(drop=True)
    all_codes = {}
    n = len(df)
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]

        for attempt in range(3):  # retry up to 3 times on failure
            try:
                codes = get_landcover_batch(batch, lat_col, lon_col)
                all_codes.update(codes)
                break
            except Exception as e:
                print(f"  Batch {i+1}/{n_batches} attempt {attempt+1} failed: {e}")
                time.sleep(5)
        else:
            print(f"  Batch {i+1}/{n_batches} FAILED after 3 attempts — skipping")

        print(f"Processed batch {i+1}/{n_batches} ({end}/{n} rows)")

    df['landcover_code'] = df.index.map(all_codes)
    df['landcover_class'] = df['landcover_code'].map(WORLDCOVER_CLASSES)
    return df

if __name__ == "__main__":
    ee.Initialize(project=PROJECT_ID)

    input_path = 'data/raw/firms_viirs_india_2months_raw.csv'
    output_path = 'data/processed/firms_with_landcover.csv'

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    df = get_landcover_for_points(df)

    print(df[['latitude', 'longitude', 'landcover_class']].head(10))
    print(f"\nMissing land-cover values: {df['landcover_class'].isna().sum()}")

    df.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")

