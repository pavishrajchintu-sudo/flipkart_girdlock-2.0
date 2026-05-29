import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pygeohash as pgh
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.replace('', np.nan, inplace=True)
    df.replace(' ', np.nan, inplace=True)
    return df

def _safe_decode_geohash(value):
    if pd.isna(value):
        return np.nan, np.nan
    try:
        lat, lon = pgh.decode(str(value))
        return lat, lon
    except Exception:
        return np.nan, np.nan


def _fit_label_encoder(series):
    encoder = LabelEncoder()
    encoder.fit(series.astype(str))
    return encoder


def _transform_label_encoder(series, encoder):
    class_map = {label: idx for idx, label in enumerate(encoder.classes_)}
    return series.astype(str).map(class_map).fillna(-1).astype(int)


def engineer_features(df, artifacts_dir='models', fit_label_encoders=False, le_road=None, le_weather=None):
    out = df.copy()

    if 'day' in out.columns:
        out['day'] = pd.to_numeric(out['day'], errors='coerce')
    if 'NumberofLanes' in out.columns:
        out['NumberofLanes'] = pd.to_numeric(out['NumberofLanes'], errors='coerce')
    if 'demand' in out.columns:
        out['demand'] = pd.to_numeric(out['demand'], errors='coerce')
    out['Temperature'] = pd.to_numeric(out['Temperature'], errors='coerce')

    decoded = out['geohash'].apply(_safe_decode_geohash)
    out['lat'] = decoded.apply(lambda pair: pair[0])
    out['lon'] = decoded.apply(lambda pair: pair[1])

    if 'timestamp' in out.columns:
        split_ts = out['timestamp'].astype(str).str.split(':', n=1, expand=True)
        hour = pd.to_numeric(split_ts[0], errors='coerce')
        minute = pd.to_numeric(split_ts[1], errors='coerce')
    else:
        hour = pd.Series(np.nan, index=out.index)
        minute = pd.Series(np.nan, index=out.index)

    out['total_minutes'] = (hour * 60) + minute
    out['hour_sin'] = np.sin(2 * np.pi * hour.fillna(0) / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour.fillna(0) / 24)
    out['min_sin'] = np.sin(2 * np.pi * minute.fillna(0) / 60)
    out['min_cos'] = np.cos(2 * np.pi * minute.fillna(0) / 60)
    out['hour_bucket'] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    out['Temperature'] = out.groupby('geohash')['Temperature'].transform(lambda x: x.fillna(x.median()))
    out['Temperature'] = out['Temperature'].fillna(out['Temperature'].median())

    out['LargeVehicles'] = out['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0}).fillna(-1)
    out['Landmarks'] = out['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(-1)
    out['RoadType'] = out['RoadType'].fillna('Unknown')
    out['Weather'] = out['Weather'].fillna('Unknown')

    if fit_label_encoders:
        os.makedirs(artifacts_dir, exist_ok=True)
        le_road = _fit_label_encoder(out['RoadType'])
        le_weather = _fit_label_encoder(out['Weather'])
        joblib.dump(le_road, os.path.join(artifacts_dir, 'le_RoadType.pkl'))
        joblib.dump(le_weather, os.path.join(artifacts_dir, 'le_Weather.pkl'))
    elif le_road is None or le_weather is None:
        raise ValueError("le_road and le_weather are required when fit_label_encoders=False")

    out['RoadType'] = _transform_label_encoder(out['RoadType'], le_road)
    out['Weather'] = _transform_label_encoder(out['Weather'], le_weather)

    out['gh4'] = out['geohash'].astype(str).str[:4]
    out['gh5'] = out['geohash'].astype(str).str[:5]

    out['lane_x_largevehicle'] = out['NumberofLanes'].fillna(0) * out['LargeVehicles']
    out['roadtype_x_hour_bucket'] = out['RoadType'] * out['hour_bucket']
    out['weather_x_hour_bucket'] = out['Weather'] * out['hour_bucket']

    out.drop(columns=['timestamp'], inplace=True, errors='ignore')
    return out, le_road, le_weather


def add_geohash_target_encodings(train_frame, apply_frame):
    train_out = train_frame.copy()
    apply_out = apply_frame.copy()

    global_demand_mean = train_out['demand'].mean()

    geohash_mean = train_out.groupby('geohash')['demand'].mean()
    gh4_mean = train_out.groupby('gh4')['demand'].mean()
    gh5_mean = train_out.groupby('gh5')['demand'].mean()

    train_out['geohash_encoded'] = train_out['geohash'].map(geohash_mean).fillna(global_demand_mean)
    apply_out['geohash_encoded'] = apply_out['geohash'].map(geohash_mean).fillna(global_demand_mean)

    train_out['gh4_encoded'] = train_out['gh4'].map(gh4_mean).fillna(global_demand_mean)
    apply_out['gh4_encoded'] = apply_out['gh4'].map(gh4_mean).fillna(global_demand_mean)

    train_out['gh5_encoded'] = train_out['gh5'].map(gh5_mean).fillna(global_demand_mean)
    apply_out['gh5_encoded'] = apply_out['gh5'].map(gh5_mean).fillna(global_demand_mean)

    drop_cols = ['geohash', 'gh4', 'gh5']
    train_out.drop(columns=drop_cols, inplace=True, errors='ignore')
    apply_out.drop(columns=drop_cols, inplace=True, errors='ignore')
    return train_out, apply_out


def get_processed_data(train_path, test_path, artifacts_dir='models'):
    """Preprocess train/test and apply train-only geohash target encodings."""
    os.makedirs(artifacts_dir, exist_ok=True)

    train_df = load_data(train_path)
    test_df = load_data(test_path)
    test_index = test_df['Index'].copy() if 'Index' in test_df.columns else None

    print("Engineering train features...")
    train_features, le_road, le_weather = engineer_features(
        train_df, artifacts_dir=artifacts_dir, fit_label_encoders=True
    )

    print("Engineering test features...")
    test_features, _, _ = engineer_features(
        test_df,
        artifacts_dir=artifacts_dir,
        fit_label_encoders=False,
        le_road=le_road,
        le_weather=le_weather
    )

    print("Applying geohash target encodings...")
    train_encoded, test_encoded = add_geohash_target_encodings(train_features, test_features)

    train_processed = train_encoded.drop(columns=['Index'], errors='ignore')
    test_processed = test_encoded.drop(columns=['demand', 'Index'], errors='ignore')
    return train_processed, test_processed, test_index
