import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

# ⚠️ Apni MAP_KEY yahan daalo
MAP_KEY = "6c60776c3cd1eb8ff79db154fbce46c3"

SENSOR = "VIIRS_SNPP_NRT"
BBOX = "68,6,97,37"   # India ka bounding box
CHUNK_SIZE = 5           # area endpoint max 5 din deta hai ek call mein
TOTAL_DAYS = 60           # ~2 mahine

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Aaj ki date se shuru karke peeche jayenge
today = datetime.utcnow().date()

all_chunks = []
current_end_date = today

num_chunks = (TOTAL_DAYS // CHUNK_SIZE) + 1

for i in range(num_chunks):
    date_str = current_end_date.strftime("%Y-%m-%d")
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{SENSOR}/{BBOX}/{CHUNK_SIZE}/{date_str}"
    )
    print(f"[{i+1}/{num_chunks}] Fetching chunk ending {date_str} ...")

    try:
        df_chunk = pd.read_csv(url)
        if not df_chunk.empty:
            all_chunks.append(df_chunk)
            print(f"    ✓ {len(df_chunk)} records")
        else:
            print("    ⚠️ Empty chunk")
    except Exception as e:
        print(f"    ✗ Error: {e}")

    # Agla chunk peeche wale 5 din ka
    current_end_date = current_end_date - timedelta(days=CHUNK_SIZE)

    # FIRMS rate limit se bachne ke liye chhota pause
    time.sleep(2)

if all_chunks:
    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates()
    out_path = RAW_DIR / "firms_viirs_india_2months_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n✓ Total combined records: {len(combined)}")
    print(f"✓ Saved: {out_path}")
    print(f"\nDate range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
else:
    print("\n✗ Koi data nahi mila kisi chunk mein")

