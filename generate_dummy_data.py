import pandas as pd, numpy as np, pathlib, random

# Set seed for reproducibility
np.random.seed(0)

n = 1000
# Generate columns
battery_start_pct = np.random.uniform(5,95,n)
# session_minutes: for highway rows we'll create large values then they'll be divided
session_minutes = np.random.uniform(10,240,n)  # base values
charger_kw = np.random.choice([3.3,7.4,22.0], n)
temp_c = np.random.uniform(15,40,n)
vehicle_age_yr = np.random.uniform(0,8,n)
queue_before = np.random.randint(0,15,n)
vehicle = np.random.choice(["car", "three_wheeler", "two_wheeler", "car.", "thr.", "three wheeler", "two.", "two wheeler"], n)
station_type = np.random.choice(["Mall", "Highway", "Residential"], n)
# kwh_billed_in_app: random numbers
kwh_billed_in_app = np.random.uniform(0,100,n)
energy_kwh = charger_kw * session_minutes / 60 * 0.8
record_id = np.arange(1, n+1)
# date column as strings
start_date = np.datetime64('2026-01-01')
dates = start_date + np.random.randint(0, 180, size=n).astype('timedelta64[D]')
# Introduce stuck sensor rows: date between 2026-02-15 and 2026-03-03 and vehicle_age_yr == 0.9
mask = (dates >= np.datetime64('2026-02-15')) & (dates <= np.datetime64('2026-03-03'))
# Ensure we have exactly 92 such rows
indices = np.where(mask)[0]
if len(indices) < 92:
    # extend by setting additional random indices to satisfy condition
    extra = 92 - len(indices)
    extra_idx = np.random.choice(np.arange(n), extra, replace=False)
    indices = np.concatenate([indices, extra_idx])
else:
    indices = indices[:92]
vehicle_age_yr[indices] = 0.9
# For highway rows, ensure session_minutes large enough so after division stays within 10-240
highway_mask = station_type == 'Highway'
# set original session_minutes for highway rows between 600 and 14400 (10*60 to 240*60)
session_minutes[highway_mask] = np.random.uniform(600, 14400, highway_mask.sum())

df = pd.DataFrame({
    'battery_start_pct': battery_start_pct,
    'session_minutes': session_minutes,
    'charger_kw': charger_kw,
    'temp_c': temp_c,
    'vehicle_age_yr': vehicle_age_yr,
    'queue_before': queue_before,
    'vehicle': vehicle,
    'station_type': station_type,
    'kwh_billed_in_app': kwh_billed_in_app,
    'energy_kwh': energy_kwh,
    'record_id': record_id,
    'date': dates.astype(str)
})

raw_dir = pathlib.Path(__file__).parent / 'data' / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)
raw_path = raw_dir / 'train.csv'
df.to_csv(raw_path, index=False)
print(f'Dummy raw data written to {raw_path}')
