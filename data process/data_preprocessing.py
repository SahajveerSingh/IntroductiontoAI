from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configurable settings
# ------------------------------------------------------------
DATASET_PATH = Path("../Scats Data October 2006.xls")
OUTPUT_DIR = Path(".")
WINDOW_SIZE = 12  # 12 x 15-minute readings = previous 3 hours

def load_and_prepare_source_data(dataset_path=DATASET_PATH):
    """Load the supplied SCATS data and validate its traffic-volume values."""
    df = pd.read_excel(
        dataset_path,
        sheet_name="Data",
        header=1,
        dtype={"SCATS Number": str}
    )
    df["SCATS Number"] = df["SCATS Number"].str.zfill(4)
    df["Date"] = pd.to_datetime(df["Date"])

    volume_columns = [f"V{i:02d}" for i in range(96)]
    numeric_volume = df[volume_columns].apply(pd.to_numeric, errors="coerce")

    checks = {
        "missing_values": int(df[volume_columns].isna().sum().sum()),
        "invalid_values": int(numeric_volume.isna().sum().sum() - df[volume_columns].isna().sum().sum()),
        "negative_values": int((numeric_volume < 0).sum().sum()),
        "duplicate_daily_rows": int(df.duplicated().sum())
    }
    df[volume_columns] = numeric_volume
    return df, volume_columns, checks

def reshape_to_time_series(df, volume_columns):
    """Convert daily V00-V95 columns into individual 15-minute timestamp records."""
    time_labels = {f"V{i:02d}": f"{i // 4:02d}:{(i % 4) * 15:02d}" for i in range(96)}
    id_columns = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE", "Date"]

    long_df = df.melt(
        id_vars=id_columns,
        value_vars=volume_columns,
        var_name="time_slot",
        value_name="traffic_flow"
    )
    long_df["time"] = long_df["time_slot"].map(time_labels)
    long_df["timestamp"] = pd.to_datetime(
        long_df["Date"].dt.strftime("%Y-%m-%d") + " " + long_df["time"]
    )
    long_df = long_df.rename(columns={
        "SCATS Number": "scats_number",
        "Location": "location",
        "NB_LATITUDE": "latitude",
        "NB_LONGITUDE": "longitude",
        "Date": "date"
    })
    return long_df[
        ["scats_number", "location", "latitude", "longitude",
         "date", "time_slot", "time", "timestamp", "traffic_flow"]
    ].sort_values(["scats_number", "location", "timestamp"]).reset_index(drop=True)

def split_and_scale(long_df):
    """Split chronologically and fit normalisation only on training records."""
    unique_dates = sorted(long_df["date"].dt.date.unique())
    train_dates = unique_dates[:21]
    validation_dates = unique_dates[21:26]
    test_dates = unique_dates[26:]

    long_df = long_df.copy()
    long_df["dataset_split"] = np.select(
        [
            long_df["date"].dt.date.isin(train_dates),
            long_df["date"].dt.date.isin(validation_dates),
            long_df["date"].dt.date.isin(test_dates)
        ],
        ["train", "validation", "test"],
        default="unassigned"
    )

    train_values = long_df.loc[long_df["dataset_split"] == "train", "traffic_flow"]
    train_min = float(train_values.min())
    train_max = float(train_values.max())
    long_df["traffic_flow_scaled"] = (long_df["traffic_flow"] - train_min) / (train_max - train_min)
    return long_df, train_min, train_max

def create_sequences(df, window_size=WINDOW_SIZE):
    """Create input/target samples without crossing location or split boundaries."""
    X_values, y_values, metadata = [], [], []
    for (split_name, site, location), group in df.groupby(
        ["dataset_split", "scats_number", "location"], sort=False
    ):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group["traffic_flow_scaled"].to_numpy(dtype=np.float32)
        for index in range(window_size, len(values)):
            X_values.append(values[index-window_size:index])
            y_values.append(values[index])
            metadata.append({
                "dataset_split": split_name,
                "scats_number": site,
                "location": location,
                "target_timestamp": group.loc[index, "timestamp"],
                "actual_traffic_flow": group.loc[index, "traffic_flow"]
            })
    return np.asarray(X_values), np.asarray(y_values), pd.DataFrame(metadata)

def main():
    df, volume_columns, checks = load_and_prepare_source_data()
    long_df = reshape_to_time_series(df, volume_columns)
    long_df, train_min, train_max = split_and_scale(long_df)
    X_all, y_all, metadata = create_sequences(long_df)

    model_arrays = {}
    for split in ["train", "validation", "test"]:
        mask = (metadata["dataset_split"] == split).to_numpy()
        model_arrays[f"X_{split}"] = X_all[mask]
        model_arrays[f"y_{split}"] = y_all[mask]

    long_df.to_csv(OUTPUT_DIR / "processed_scats_time_series_with_split_scaled.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "sequence_metadata_window12.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "model_ready_sequences_window12.npz", **model_arrays)

    print("Preprocessing complete.")
    print("Validation checks:", checks)
    for key, value in model_arrays.items():
        print(key, value.shape)
    print("Scaling fitted on training data only:", train_min, "to", train_max)

if __name__ == "__main__":
    main()
