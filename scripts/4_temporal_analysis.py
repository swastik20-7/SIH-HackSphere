import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("data/processed/firms_processed.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

if not CSV.exists():
    raise SystemExit(f"✗ File nahi mila: {CSV}. Pehle 2_clean_data.py run karo.")

df = pd.read_csv(CSV)

# acq_time FIRMS mein HHMM format (UTC) mein hoti hai, e.g. 830, 1345
df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
df["hour"] = df["acq_time"].str[:2].astype(int)

df["acq_date"] = pd.to_datetime(df["acq_date"])
df["day"] = df["acq_date"].dt.day
df["month"] = df["acq_date"].dt.month
df["dayofweek"] = df["acq_date"].dt.dayofweek

hourly_counts = df.groupby("hour").size()
daily_counts = df.groupby("day").size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(hourly_counts.index, hourly_counts.values, color="orange", edgecolor="black")
axes[0].set_xlabel("Hour of Day (UTC)")
axes[0].set_ylabel("Number of Detections")
axes[0].set_title("Diurnal Pattern of Thermal Hotspots", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(daily_counts.index, daily_counts.values, color="red", edgecolor="black")
axes[1].set_xlabel("Day of Month")
axes[1].set_ylabel("Number of Detections")
axes[1].set_title("Daily Variation", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = OUT / "firms_temporal_pattern.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✓ Temporal plots saved: {out_path}")
plt.show()

# Insight note:
# Gas flare -> roughly flat hourly distribution (burns 24/7)
# Crop burning -> peak during daytime hours (~10 AM - 2 PM local time)
print("\nNote: Gas flares = flat hourly spread. Crop burning = daytime peak.")