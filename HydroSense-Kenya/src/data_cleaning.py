"""
data_cleaning.py – HydroSense-Kenya
ICS 2207 Scientific Computing | Level 4
"""

import pandas as pd
import numpy as np


def load_datasets(base_path="../data/raw"):
    weather = pd.read_csv(f"{base_path}/weather_daily.csv", na_values=["NA", ""])
    soil    = pd.read_csv(f"{base_path}/soil_sensor_data.csv", na_values=["NA", ""])
    params  = pd.read_csv(f"{base_path}/crop_zone_parameters.csv", na_values=["NA", ""])
    weather['date'] = pd.to_datetime(weather['date'])
    soil['timestamp'] = pd.to_datetime(soil['timestamp'])
    soil['date'] = soil['timestamp'].dt.normalize()
    return weather, soil, params


def detect_outliers_iqr(series):
    """Return boolean mask of outliers using IQR method."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)


def clean_weather(df):
    cleaned = df.copy()
    report = []

    # 1. Fill missing rainfall with 0 (missing sensor = no rain recorded)
    missing_rain = cleaned['rainfall_mm'].isna().sum()
    cleaned['rainfall_mm'] = cleaned['rainfall_mm'].fillna(0.0)
    report.append(f"rainfall_mm: filled {missing_rain} NA with 0.0")

    # 2. Fill missing humidity with column median
    missing_hum = cleaned['humidity_pct'].isna().sum()
    cleaned['humidity_pct'] = cleaned['humidity_pct'].fillna(cleaned['humidity_pct'].median())
    report.append(f"humidity_pct: filled {missing_hum} NA with median")

    # 3. Cap temperature outlier (45.8°C is physically implausible for Nairobi March)
    temp_outliers = detect_outliers_iqr(cleaned['temperature_c'])
    cleaned.loc[temp_outliers, 'temperature_c'] = cleaned['temperature_c'].median()
    report.append(f"temperature_c: replaced {temp_outliers.sum()} outlier(s) with median")

    # 4. Cap extreme rainfall outlier (85 mm is a 1-in-50-year event; flag but keep)
    rain_outliers = detect_outliers_iqr(cleaned['rainfall_mm'])
    cleaned['rainfall_outlier_flag'] = rain_outliers
    report.append(f"rainfall_mm: flagged {rain_outliers.sum()} outlier(s) (kept, flagged)")

    for r in report:
        print(f"  [weather] {r}")
    return cleaned


def clean_soil(df):
    cleaned = df.copy()
    report = []

    # 1. Fill missing soil_moisture_pct per zone using zone median
    for zone in cleaned['zone_id'].unique():
        mask = cleaned['zone_id'] == zone
        missing = cleaned.loc[mask, 'soil_moisture_pct'].isna().sum()
        med = cleaned.loc[mask, 'soil_moisture_pct'].median()
        cleaned.loc[mask & cleaned['soil_moisture_pct'].isna(), 'soil_moisture_pct'] = med
        if missing:
            report.append(f"soil_moisture_pct [{zone}]: filled {missing} NA with zone median {med:.2f}")

    # 2. Fix implausible soil_moisture (Zone_B 8.5% on 2026-03-25 – sensor fault)
    low_mask = cleaned['soil_moisture_pct'] < 10
    cleaned.loc[low_mask, 'soil_moisture_pct'] = np.nan
    for zone in cleaned['zone_id'].unique():
        mask = cleaned['zone_id'] == zone
        cleaned.loc[mask, 'soil_moisture_pct'] = cleaned.loc[mask, 'soil_moisture_pct'].interpolate(method='linear')
    report.append(f"soil_moisture_pct: replaced {low_mask.sum()} implausibly low value(s) with linear interpolation")

    # 3. Fix tank_level outlier (Zone_C 9900 L on 2026-03-14 – impossible given 5000 L tank)
    tank_outliers = cleaned['tank_level_liters'] > 6000
    cleaned.loc[tank_outliers, 'tank_level_liters'] = np.nan
    cleaned['tank_level_liters'] = cleaned['tank_level_liters'].interpolate(method='linear')
    report.append(f"tank_level_liters: replaced {tank_outliers.sum()} outlier(s) with interpolation")

    # 4. Flag CHECK sensor rows
    check_rows = (cleaned['sensor_status'] == 'CHECK')
    cleaned['sensor_fault_flag'] = check_rows
    report.append(f"sensor_status: flagged {check_rows.sum()} CHECK row(s)")

    # 5. Flag pump_flow = 0 (possible pump failure)
    zero_flow = (cleaned['pump_flow_lpm'] == 0.0)
    cleaned['pump_fault_flag'] = zero_flow
    report.append(f"pump_flow_lpm: flagged {zero_flow.sum()} zero-flow row(s)")

    for r in report:
        print(f"  [soil]    {r}")
    return cleaned


def save_cleaned(weather_clean, soil_clean, path="../data/processed/cleaned_irrigation_dataset.csv"):
    """Merge cleaned weather and soil data into one flat dataset and save."""
    soil_wide = soil_clean.pivot_table(
        index='date', columns='zone_id',
        values='soil_moisture_pct', aggfunc='mean'
    ).reset_index()
    soil_wide.columns = ['date'] + [f'sm_{z.lower().replace("_","")}_pct' for z in soil_wide.columns[1:]]

    merged = weather_clean.merge(soil_wide, on='date', how='left')
    merged.to_csv(path, index=False)
    print(f"\nCleaned dataset saved → {path}  ({len(merged)} rows, {len(merged.columns)} columns)")
    return merged
