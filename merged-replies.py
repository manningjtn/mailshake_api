from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any
import gzip

import boto3
import pandas as pd
from bs4 import BeautifulSoup

BUCKET = "mailshake-analysis"
SOURCE_PREFIX = "activity-replies/"
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_KEY = f"merged/merged-replies/replies_{timestamp}.csv"

INCLUDED_FIELDS = [
    "team_id",
    "object",
    "id",
    "actiondate",
    "type",
    "subject",
    "plaintextbody",
    "recipient_object",
    "recipient_id",
    "recipient_emailaddress",
    "recipient_created",
    "recipient_ispaused",
    "recipient_contactid",
    "recipient_fields_link",
    "recipient_fields_position",
    "recipient_fields_date_applied",
    "recipient_fields_account",
    "recipient_fields_linkedinurl",
    "campaign_object",
    "campaign_id",
    "campaign_title",
    "campaign_wizardstatus",
    "parent_object",
    "parent_id",
    "parent_type",
    "parent_message_object",
    "parent_message_id",
    "parent_message_type",
    "parent_message_subject",
    "parent_message_replytoid",
    "from_object",
    "from_address",
]

REQUIRED_FIELDS = ["team_id", "object", "id"]

OUTPUT_FIELDS = INCLUDED_FIELDS


# ============================================================
# S3 HELPERS
# ============================================================

def get_s3_client():
    return boto3.client("s3")


def is_supported_csv_key(key: str) -> bool:
    """Return True for .csv and .csv.gz files."""
    key_lower = key.lower()
    return key_lower.endswith(".csv") or key_lower.endswith(".csv.gz")


def list_source_files(s3, bucket: str, prefix: str, output_key: str) -> list[str]:
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
    obj = s3.get_object(Bucket=bucket, Key=key)

    # force strings to prevent schema drift
    if key.lower().endswith(".csv.gz"):
        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as gz:
            return pd.read_csv(gz, dtype=str, keep_default_na=False)

    return pd.read_csv(obj["Body"], dtype=str, keep_default_na=False)


def write_csv_to_s3(s3, df: pd.DataFrame, bucket: str, key: str) -> None:
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
    )


# ============================================================
# SCHEMA + CLEANING
# ============================================================

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # normalize column names
    result.columns = [str(col).strip().lower() for col in result.columns]

    # missing required → skip file (0 rows)
    missing_required = [col for col in REQUIRED_FIELDS if col not in result.columns]
    if missing_required:
        print(f"Skipping file — missing required columns: {missing_required}")
        return pd.DataFrame(columns=INCLUDED_FIELDS)

    # add missing fields
    for col in INCLUDED_FIELDS:
        if col not in result.columns:
            result[col] = pd.NA

    # enforce schema + order
    result = result[INCLUDED_FIELDS]

    # normalize strings
    for col in INCLUDED_FIELDS:
        result[col] = result[col].astype("string").str.strip()

    result = result.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # normalize object
    result["object"] = result["object"].str.lower()

    # drop bad rows
    before = len(result)
    result = result.dropna(subset=REQUIRED_FIELDS).reset_index(drop=True)
    dropped = before - len(result)

    if dropped > 0:
        print(f"Dropped {dropped} invalid rows")

    return result


def strip_html(value: Any) -> Any:
    if pd.isna(value):
        return value
    return BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)


def clean_html(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in result.columns:
        result[col] = result[col].map(strip_html)

    return result


def remove_mime_fragments(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"=\?.*?\?=", "", regex=True)


# ============================================================
# TRANSFORM
# ============================================================

def transform_merged_replies(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)

    merged = merged.drop_duplicates().reset_index(drop=True)

    merged = merged[merged["object"] == "reply"].reset_index(drop=True)

    merged = clean_html(merged)

    merged = remove_mime_fragments(merged)

    merged = merged[OUTPUT_FIELDS]

    return merged


# ============================================================
# CLEANUP
# ============================================================

def empty_s3_prefix(s3, bucket: str, prefix: str) -> None:
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
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})
                objects_to_delete = []

    if objects_to_delete:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})

    print(f"Deleted CSV/CSV.GZ files from s3://{bucket}/{prefix}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    s3 = get_s3_client()

    files = list_source_files(s3, BUCKET, SOURCE_PREFIX, OUTPUT_KEY)

    if not files:
        print("No source CSV or CSV.GZ files found")
        return

    dfs: list[pd.DataFrame] = []

    for key in files:
        print(f"Reading {key}")

        df = read_csv_from_s3(s3, BUCKET, key)
        df = enforce_schema(df)

        if not df.empty:
            dfs.append(df)
        else:
            print(f"Skipped {key}")

    if not dfs:
        print("No valid data to process")
        return

    merged = transform_merged_replies(dfs)

    if merged.empty:
        print("No reply rows found")
        return

    write_csv_to_s3(s3, merged, BUCKET, OUTPUT_KEY)

    print(f"Saved merged file to s3://{BUCKET}/{OUTPUT_KEY}")

    empty_s3_prefix(s3, BUCKET, SOURCE_PREFIX)


if __name__ == "__main__":
    main()