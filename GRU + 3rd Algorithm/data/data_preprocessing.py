from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Configurable settings
# ------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "Scats Data October 2006.xls"
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
    """Convert daily V00-V95 columns into individual 15-minute timestamp records.

    Duplicate site-location-timestamp records (caused by source rows that share
    the same SCATS Number, Location and Date but are not exact row duplicates)
    are resolved by averaging their traffic-flow values so that each
    (scats_number, location, timestamp) combination is unique.
    """
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
    long_df = long_df[
        ["scats_number", "location", "latitude", "longitude",
         "date", "time_slot", "time", "timestamp", "traffic_flow"]
    ].sort_values(["scats_number", "location", "timestamp"]).reset_index(drop=True)

    # Detect and resolve duplicate (scats_number, location, timestamp) records.
    dup_key = ["scats_number", "location", "timestamp"]
    n_duplicates = int(long_df.duplicated(subset=dup_key).sum())
    if n_duplicates > 0:
        print(f"[reshape] {n_duplicates} duplicate site-location-timestamp records found. "
              "Resolving by averaging traffic-flow values.")
        # Keep first occurrence of non-traffic columns; average traffic_flow.
        meta_cols = ["scats_number", "location", "latitude", "longitude",
                     "date", "time_slot", "time", "timestamp"]
        long_df = (
            long_df
            .groupby(dup_key, sort=False)
            .agg({**{c: "first" for c in meta_cols if c not in dup_key},
                  "traffic_flow": "mean"})
            .reset_index()
        )[["scats_number", "location", "latitude", "longitude",
           "date", "time_slot", "time", "timestamp", "traffic_flow"]]
        long_df = long_df.sort_values(["scats_number", "location", "timestamp"]).reset_index(drop=True)

    return long_df, n_duplicates

def split_and_scale(long_df):
    """Split chronologically and fit normalisation only on training records.

    Notes
    -----
    - Scaling parameters are derived from training data only to prevent
      information leakage into validation and test sets.
    - Validation and test splits may contain traffic-flow values that exceed
      the training maximum, producing scaled values above 1.0.  This is
      expected and correct behaviour; models must tolerate inputs outside
      the [0, 1] range seen during training.
    - Sequences do not restart at midnight within a split: the sliding window
      in create_sequences operates continuously over each
      (dataset_split, scats_number, location) group, so the first window of
      a new day will include observations from the previous day.  This is
      intentional for time-series modelling.
    """
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

    # Warn if any split contains values outside the training scaling range.
    for split in ["validation", "test"]:
        split_max = long_df.loc[long_df["dataset_split"] == split, "traffic_flow_scaled"].max()
        if split_max > 1.0:
            print(f"[split_and_scale] Note: '{split}' scaled values reach {split_max:.4f} "
                  f"(above 1.0) because its traffic-flow maximum exceeds the training maximum "
                  f"of {train_max}. This is expected; no action required.")

    return long_df, train_min, train_max

def create_sequences(df, window_size=WINDOW_SIZE):
    """Create input/target samples without crossing location or split boundaries.

    The sliding window operates continuously over each (dataset_split,
    scats_number, location) group.  Windows are not restarted at midnight,
    so a window beginning near the start of a new calendar day will include
    observations from the previous day.  This is intentional: traffic flow
    is a continuous signal and an artificial gap at midnight would discard
    valid context.
    """
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
    long_df, n_duplicates = reshape_to_time_series(df, volume_columns)
    checks["duplicate_site_location_timestamp_records"] = n_duplicates
    checks["duplicate_resolution_method"] = "averaged" if n_duplicates > 0 else "none_required"

    long_df, train_min, train_max = split_and_scale(long_df)
    X_all, y_all, metadata = create_sequences(long_df)

    model_arrays = {}
    for split in ["train", "validation", "test"]:
        mask = (metadata["dataset_split"] == split).to_numpy()
        model_arrays[f"X_{split}"] = X_all[mask]
        model_arrays[f"y_{split}"] = y_all[mask]

    # Build split summary with clean date-only formatting.
    split_summary = (
        long_df.groupby("dataset_split", as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            time_series_rows=("traffic_flow", "size"),
            minimum_flow=("traffic_flow", "min"),
            maximum_flow=("traffic_flow", "max"),
        )
    )
    split_summary["first_date"] = split_summary["first_date"].dt.date
    split_summary["last_date"] = split_summary["last_date"].dt.date
    seq_counts = (
        metadata.groupby("dataset_split")
        .size()
        .rename(f"sequence_samples_window{WINDOW_SIZE}")
        .reset_index()
    )
    split_summary = split_summary.merge(seq_counts, on="dataset_split")

    long_df.to_csv(OUTPUT_DIR / "processed_scats_time_series_with_split_scaled.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / "sequence_metadata_window12.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "model_ready_sequences_window12.npz", **model_arrays)
    split_summary.to_csv(OUTPUT_DIR / "chronological_split_summary.csv", index=False)

    # Build quality summary CSV.
    quality_rows = [
        ("Original daily records loaded", len(df)),
        ("Unique SCATS sites", df["SCATS Number"].nunique()),
        ("Recorded location/direction entries",
         df[["SCATS Number", "Location"]].drop_duplicates().shape[0]),
        ("15-minute traffic volume columns", len(volume_columns)),
        ("Missing traffic-flow values", checks["missing_values"]),
        ("Invalid non-numeric traffic-flow values", checks["invalid_values"]),
        ("Negative traffic-flow values", checks["negative_values"]),
        ("Duplicate original daily records", checks["duplicate_daily_rows"]),
        ("Duplicate site-location-timestamp records (post-reshape)", n_duplicates),
        ("Duplicate resolution method", checks["duplicate_resolution_method"]),
        ("Time-series records created (after deduplication)", len(long_df)),
        ("Normalisation fitted using training data only", "Yes"),
        ("Training traffic minimum used for scaling", train_min),
        ("Training traffic maximum used for scaling", train_max),
        (f"Note: val/test scaled values may exceed 1.0",
         "Expected — val/test max flow can exceed training max"),
        ("Sequence window length used for demonstration",
         f"{WINDOW_SIZE} intervals (3 hours)"),
        ("Sequences restart at midnight", "No — continuous within each split/site/location group"),
    ]
    pd.DataFrame(quality_rows, columns=["Processing Check", "Result"]).to_csv(
        OUTPUT_DIR / "data_quality_and_processing_summary.csv", index=False
    )

    print("Preprocessing complete.")
    print("Validation checks:", checks)
    for key, value in model_arrays.items():
        print(key, value.shape)
    print("Scaling fitted on training data only:", train_min, "to", train_max)
    print(split_summary.to_string(index=False))

if __name__ == "__main__":
    main()
