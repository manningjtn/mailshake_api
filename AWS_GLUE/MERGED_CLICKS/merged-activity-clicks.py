"""
Mailshake Clicks Merge Job

Purpose:
    Merge all team-level Mailshake clicks CSV files from S3 into a single
    consolidated dataset, deduplicate rows, and write the final merged CSV
    back to S3.

Source:
    s3://mailshake-analysis/activity-clicks/*.csv
    s3://mailshake-analysis/activity-clicks/*.csv.gz

Destination:
    s3://mailshake-analysis/merged/merged-clicks/clicks.csv

Notes:
    - This script skips the final merged output file so it does not re-read its
      own result.
    - Only .csv and .csv.gz files are deleted from the source prefix after a successful run.
    - Output schema is explicitly controlled to prevent drift.
    - Files missing required columns are kept; missing required fields are added as NULL.
    - Rows missing required values are dropped.
"""

from __future__ import annotations

import csv
import gzip
from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd

BUCKET = "mailshake-analysis"
SOURCE_PREFIX = "activity-clicks/"
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_KEY = f"merged/merged-clicks/clicks_{timestamp}.csv"

SOURCE_PREFIX_TO_EMPTY = "activity-clicks/"


# ============================================================
# SCHEMA CONTROL
# ============================================================

INCLUDED_FIELDS = [
    "team_id",
    "object",
    "id",
    "link",
    "actiondate",
    "isduplicate",
    "recipient_object",
    "recipient_id",
    "recipient_emailaddress",
    "recipient_fullname",
    "recipient_created",
    "recipient_ispaused",
    "recipient_contactid",
    "recipient_first",
    "recipient_last",
    "recipient_fields_city",
    "recipient_fields_last",
    "recipient_fields_first",
    "recipient_fields_state",
    "recipient_fields_full_name",
    "recipient_fields_phonenumber",
    "recipient_fields_linkedinurl",
    "campaign_object",
    "campaign_id",
    "campaign_title",
    "parent_object",
    "parent_id",
    "parent_type",
    "parent_message_object",
    "parent_message_id",
    "parent_message_type",
    "parent_message_subject",
    "parent_message_replytoid",
    "recipient_fields_phone_2_com",
]

REQUIRED_FIELDS = [
    "team_id",
    "object",
    "id",
]

OUTPUT_FIELDS = INCLUDED_FIELDS


# ============================================================
# S3 HELPERS
# ============================================================

def get_s3_client():
    """
    Create and return an S3 client.
    """
    return boto3.client("s3")


def list_source_files(s3, bucket: str, prefix: str, output_key: str) -> list[str]:
    """
    List source CSV files to merge.

    Excludes the final merged output file so the script does not accidentally
    read its own output.

    Supports both .csv and .csv.gz inputs.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files: list[str] = []

    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]

            if key == output_key:
                continue

            if key.lower().endswith((".csv", ".csv.gz")):
                files.append(key)

    print(f"Discovered {len(files)} source file(s)")
    for key in files:
        print(f" - {key}")

    return files


def read_csv_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    """
    Read a CSV or CSV.GZ file from S3 into a pandas DataFrame.

    Read all values as strings to prevent schema drift caused by pandas dtype
    inference across files.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()

    if key.lower().endswith(".csv.gz"):
        print(f"Decompressing gz file: {key}")
        raw = gzip.decompress(raw)
    else:
        print(f"Reading csv file: {key}")

    data = raw.decode("utf-8")

    return pd.read_csv(
        StringIO(data),
        sep=",",
        quotechar='"',
        engine="python",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
    )


def write_csv_to_s3(s3, df: pd.DataFrame, bucket: str, key: str) -> None:
    """
    Write a pandas DataFrame to S3 as CSV.
    """
    csv_buffer = StringIO()
    df.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
    )
    csv_buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
        ServerSideEncryption="AES256",
    )


# ============================================================
# TRANSFORMATION HELPERS
# ============================================================

def enforce_schema(df: pd.DataFrame, source_key: str | None = None) -> pd.DataFrame:
    """
    Force DataFrame into the expected schema.

    Behavior:
        - Never skip the whole file just because required columns are missing.
        - Add missing required/included fields as NULL.
        - Drop unexpected columns.
        - Normalize values as trimmed strings.
        - Drop only rows missing required values.
    """
    result = df.copy()

    # Normalize column names
    result.columns = [str(col).strip().lower() for col in result.columns]

    missing_required = [col for col in REQUIRED_FIELDS if col not in result.columns]
    if missing_required:
        location = f" in {source_key}" if source_key else ""
        print(
            f"Missing required columns{location}: {missing_required} "
            f"— adding them as nulls and continuing"
        )

    # Add missing included fields
    for col in INCLUDED_FIELDS:
        if col not in result.columns:
            result[col] = pd.NA

    # Drop anything not explicitly allowed and enforce column order
    result = result[INCLUDED_FIELDS]

    # Normalize all fields to string and trim whitespace
    for col in INCLUDED_FIELDS:
        result[col] = result[col].astype("string").str.strip()

    # Clean common empty-like values
    result = result.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "null": pd.NA,
        "NULL": pd.NA,
    })

    # Normalize object values for strict filtering
    if "object" in result.columns:
        result["object"] = result["object"].str.lower()

    # Drop rows missing required values
    before_count = len(result)
    result = result.dropna(subset=REQUIRED_FIELDS).reset_index(drop=True)
    dropped = before_count - len(result)

    if dropped > 0:
        location = f" from {source_key}" if source_key else ""
        print(
            f"Dropped {dropped} invalid rows{location} "
            f"(missing one of {REQUIRED_FIELDS})"
        )

    return result


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames into one merged DataFrame.
    """
    if not dfs:
        raise ValueError("No valid source CSV files found to merge.")

    return pd.concat(dfs, ignore_index=True, sort=False)


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Prefer deduping by team_id + id when available.
    """
    if "team_id" in df.columns and "id" in df.columns:
        return df.drop_duplicates(subset=["team_id", "id"]).reset_index(drop=True)

    if "id" in df.columns:
        return df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    return df.drop_duplicates().reset_index(drop=True)


def filter_click_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows representing click records.
    """
    return df[df["object"] == "click"].reset_index(drop=True)


def transform_merged_clicks(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Apply the full transformation pipeline for merged clicks data.

    Steps:
        1. Concatenate all normalized source files
        2. Remove duplicate rows
        3. Keep only click rows
        4. Enforce final output column order
    """
    merged = merge_dataframes(dfs)
    merged = deduplicate_rows(merged)
    merged = filter_click_rows(merged)
    merged = merged[OUTPUT_FIELDS]
    return merged


# ============================================================
# SOURCE FOLDER CLEANUP
# ============================================================

def empty_s3_prefix(s3, bucket: str, prefix: str) -> None:
    """
    Delete only .csv and .csv.gz files under an S3 prefix.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects_to_delete = []

    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]

            if not key.lower().endswith((".csv", ".csv.gz")):
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
    """
    Empty this script's source prefix after successful processing.
    """
    empty_s3_prefix(s3, BUCKET, SOURCE_PREFIX_TO_EMPTY)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main execution flow.

    Steps:
        - list all source CSV / CSV.GZ files in the clicks prefix
        - read each file from S3
        - normalize each file to the required schema
        - keep only non-empty normalized DataFrames
        - merge all files together
        - keep only click rows
        - deduplicate rows
        - write the final merged CSV back to S3
        - delete only source CSV / CSV.GZ files from the clicks prefix
    """
    s3 = get_s3_client()

    files = list_source_files(
        s3=s3,
        bucket=BUCKET,
        prefix=SOURCE_PREFIX,
        output_key=OUTPUT_KEY,
    )

    if not files:
        print(f"No source files found under s3://{BUCKET}/{SOURCE_PREFIX}")
        return

    dfs: list[pd.DataFrame] = []

    for key in files:
        print(f"Reading {key}")
        df = read_csv_from_s3(s3, BUCKET, key)
        df = enforce_schema(df, source_key=key)

        if not df.empty:
            dfs.append(df)
        else:
            print(f"Skipped {key} — no valid rows after row-level schema filtering")

    if not dfs:
        print("No valid rows found in any source files. Nothing to merge.")
        return

    merged = transform_merged_clicks(dfs)

    if merged.empty:
        print("Merged output has 0 valid click rows. Nothing will be written.")
        return

    write_csv_to_s3(
        s3=s3,
        df=merged,
        bucket=BUCKET,
        key=OUTPUT_KEY,
    )

    print(f"Saved merged file to: s3://{BUCKET}/{OUTPUT_KEY}")

    print("Emptying source folder...")
    empty_source_folder(s3)


if __name__ == "__main__":
    main()
