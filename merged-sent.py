from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import gzip

import boto3
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BUCKET = "mailshake-analysis"
INPUT_PREFIX = "activity-sent/"

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_KEY = f"merged/merged-sent/sent_{timestamp}.csv"

INCLUDED_FIELDS = [
    "team_id",
    "object",
    "id",
    "actiondate",
    "recipient_object",
    "recipient_id",
    "recipient_emailaddress",
    "recipient_fullname",
    "recipient_created",
    "recipient_ispaused",
    "recipient_contactid",
    "recipient_first",
    "recipient_last",
    "campaign_object",
    "campaign_id",
    "campaign_title",
    "campaign_wizardstatus",
    "message_object",
    "message_id",
    "message_subject",
    "message_body",
    "message_plaintextbody",
    "message_rawbody",
    "message_replytoid",
]

REQUIRED_FIELDS = ["team_id", "object", "id"]

OUTPUT_FIELDS = INCLUDED_FIELDS


# =========================================================
# S3 HELPERS
# =========================================================

def get_s3_client():
    return boto3.client("s3")


def is_supported_csv_key(key: str) -> bool:
    """Return True for .csv and .csv.gz files."""
    key_lower = key.lower()
    return key_lower.endswith(".csv") or key_lower.endswith(".csv.gz")


def list_csv_files(s3, bucket: str, prefix: str, output_key: str) -> list[str]:
    """List all CSV / CSV.GZ files in an S3 prefix, excluding the merged output file."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files: list[str] = []

    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]
            if is_supported_csv_key(key) and key != output_key:
                files.append(key)

    return files


def read_csv_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    """Read a .csv or .csv.gz file from S3 into a DataFrame."""
    obj = s3.get_object(Bucket=bucket, Key=key)

    if key.lower().endswith(".csv.gz"):
        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as gz:
            return pd.read_csv(gz, dtype=str, keep_default_na=False)

    return pd.read_csv(obj["Body"], dtype=str, keep_default_na=False)


def write_csv_to_s3(s3, df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write DataFrame to S3 as CSV."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
        ServerSideEncryption="AES256",
    )


# =========================================================
# TRANSFORM HELPERS
# =========================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return result


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force DataFrame into the expected schema.

    Behavior:
        - If required columns are missing, return an empty DataFrame so the file
          contributes 0 rows.
        - Add missing included fields as null.
        - Drop unexpected columns.
        - Normalize values as trimmed strings.
        - Drop rows missing required values.
    """
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]

    missing_required = [col for col in REQUIRED_FIELDS if col not in result.columns]
    if missing_required:
        print(f"Skipping file — missing required columns: {missing_required}")
        return pd.DataFrame(columns=INCLUDED_FIELDS)

    for col in INCLUDED_FIELDS:
        if col not in result.columns:
            result[col] = pd.NA

    result = result[INCLUDED_FIELDS]

    for col in INCLUDED_FIELDS:
        result[col] = result[col].astype("string").str.strip()

    result = result.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    result["object"] = result["object"].str.lower()

    before_count = len(result)
    result = result.dropna(subset=REQUIRED_FIELDS).reset_index(drop=True)
    dropped = before_count - len(result)

    if dropped > 0:
        print(f"Dropped {dropped} invalid rows (missing team_id, object, or id)")

    return result


def normalize_datetime_column(series: pd.Series) -> pd.Series:
    """Normalize a datetime-like column to a stable string format."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate sent rows."""
    if "team_id" in df.columns and "id" in df.columns:
        return df.drop_duplicates(subset=["team_id", "id"], keep="first").reset_index(drop=True)

    if "id" in df.columns:
        return df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    return df.drop_duplicates().reset_index(drop=True)


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple DataFrames into one merged DataFrame."""
    if not dfs:
        raise ValueError("No valid CSV files found to merge.")

    return pd.concat(dfs, ignore_index=True)


def transform_data(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge, sort, deduplicate, filter, and enforce schema."""
    df = merge_dataframes(dfs)

    if "actiondate" in df.columns:
        df["actiondate"] = normalize_datetime_column(df["actiondate"])
        sortable = pd.to_datetime(df["actiondate"], errors="coerce")
        df = df.assign(_sort_actiondate=sortable).sort_values("_sort_actiondate", ascending=True)
        df = df.drop(columns=["_sort_actiondate"])

    df = deduplicate_rows(df)

    if "object" in df.columns:
        df["object"] = df["object"].astype(str).str.strip().str.lower()
        df = df[df["object"] == "sent-message"].reset_index(drop=True)

    df = df[OUTPUT_FIELDS]
    return df


# =========================================================
# SOURCE FOLDER CLEANUP
# =========================================================

SOURCE_PREFIX_TO_EMPTY = "activity-sent/"


def empty_s3_prefix(s3, bucket: str, prefix: str) -> None:
    """Delete only .csv and .csv.gz files under an S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects_to_delete = []

    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]

            if not is_supported_csv_key(key):
                continue

            objects_to_delete.append({"Key": key})

            if len(objects_to_delete) == 1000:
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objects_to_delete},
                )
                objects_to_delete = []

    if objects_to_delete:
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects_to_delete},
        )

    print(f"Deleted CSV/CSV.GZ files from s3://{bucket}/{prefix}")


def empty_source_folder(s3) -> None:
    """Wrapper to clear the activity-sent folder."""
    empty_s3_prefix(s3, BUCKET, SOURCE_PREFIX_TO_EMPTY)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """End-to-end pipeline execution."""
    s3 = get_s3_client()

    print("Listing source files...")
    files = list_csv_files(s3, BUCKET, INPUT_PREFIX, OUTPUT_KEY)

    if not files:
        print("No CSV or CSV.GZ files found in source location.")
        return

    print(f"Found {len(files)} files")

    print("Reading files...")
    dfs: list[pd.DataFrame] = []

    for key in files:
        print(f"Reading {key}")
        df = read_csv_from_s3(s3, BUCKET, key)
        df = standardize_columns(df)
        df = enforce_schema(df)

        if not df.empty:
            dfs.append(df)
        else:
            print(f"Skipped {key} — no valid rows after schema enforcement")

    if not dfs:
        print("No valid data to process.")
        return

    print("Transforming data...")
    merged = transform_data(dfs)

    if merged.empty:
        print("Merged output has 0 valid sent rows. Nothing will be written.")
        return

    print("Writing output...")
    write_csv_to_s3(s3=s3, df=merged, bucket=BUCKET, key=OUTPUT_KEY)

    print(f"Saved merged file to: s3://{BUCKET}/{OUTPUT_KEY}")
    print(f"Total rows: {len(merged)}")

    print("Emptying source folder...")
    empty_source_folder(s3)


if __name__ == "__main__":
    main()