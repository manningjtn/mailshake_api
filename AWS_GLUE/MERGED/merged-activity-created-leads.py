from __future__ import annotations

import ast
import csv
import gzip
from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BUCKET = "mailshake-analysis"
SOURCE_PREFIX = "activity-created-leads/"

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_KEY = f"merged/merged-created-leads/created_leads_{timestamp}.csv"

INCLUDED_FIELDS = [
    "team_id",
    "object",
    "id",
    "recipient_object",
    "recipient_id",
    "recipient_emailaddress",
    "recipient_fullname",
    "recipient_created",
    "recipient_ispaused",
    "recipient_contactid",
    "recipient_first",
    "recipient_last",
    "recipient_fields_id",
    "recipient_fields_emailaddress",
    "recipient_fields_first",
    "recipient_fields_last",
    "recipient_fields_linkedinurl",
    "recipient_fields_position",
    "recipient_fields_date_applied",
    "recipient_fields_account",
    "campaign_object",
    "campaign_id",
    "campaign_title",
    "campaign_wizardstatus",
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
    return boto3.client("s3")


def list_source_files(s3, bucket: str, prefix: str, output_key: str) -> list[str]:
    """
    List source CSV files to merge, excluding the final merged output file.
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


def is_nonempty_object(s3, bucket: str, key: str) -> bool:
    """Check whether an S3 object exists and has nonzero content length."""
    head = s3.head_object(Bucket=bucket, Key=key)
    return head.get("ContentLength", 0) > 0


def read_csv_from_s3(s3, bucket: str, key: str) -> pd.DataFrame | None:
    """
    Read a CSV or CSV.GZ file from S3 into a pandas DataFrame.
    """
    if not is_nonempty_object(s3, bucket, key):
        return None

    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()

    try:
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
    except (pd.errors.EmptyDataError, gzip.BadGzipFile):
        return None


def write_csv_to_s3(s3, df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a pandas DataFrame to S3 as CSV."""
    buf = StringIO()
    df.to_csv(
        buf,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
    )
    buf.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )


# ============================================================
# TRANSFORMATION HELPERS
# ============================================================

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


def safe_parse_dict(value):
    """Safely parse a stringified Python dict."""
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return {}
    return value


def flatten_recipient_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten recipient.fields / recipient_fields into individual columns."""
    result = df.copy()

    candidate_columns = ["recipient.fields", "recipient_fields"]
    source_column = None

    for col in candidate_columns:
        if col in result.columns:
            source_column = col
            break

    if source_column is None:
        return result

    working = result.copy()
    working[source_column] = working[source_column].apply(safe_parse_dict)

    parsed = working[source_column].apply(lambda v: v if isinstance(v, dict) else {})
    flat = pd.json_normalize(parsed)

    flat.columns = [
        f"{source_column}_{str(col).strip().replace('.', '_').replace(' ', '_').lower()}"
        for col in flat.columns
    ]

    working = working.drop(columns=[source_column]).reset_index(drop=True)
    flat = flat.reset_index(drop=True)

    merged = pd.concat([working, flat], axis=1)
    return standardize_columns(merged)


def enforce_schema(df: pd.DataFrame, source_key: str | None = None) -> pd.DataFrame:
    """
    Force DataFrame into the expected schema.

    Behavior:
        - Never skip the whole file just because required columns are missing.
        - Add missing included fields as null.
        - Drop unexpected columns.
        - Normalize values as trimmed strings.
        - Drop only rows missing required values.
    """
    result = df.copy()

    result.columns = [str(col).strip().lower() for col in result.columns]

    missing_required = [col for col in REQUIRED_FIELDS if col not in result.columns]
    if missing_required:
        location = f" in {source_key}" if source_key else ""
        print(
            f"Missing required columns{location}: {missing_required} "
            f"— adding them as nulls and continuing"
        )

    for col in INCLUDED_FIELDS:
        if col not in result.columns:
            result[col] = pd.NA

    result = result[INCLUDED_FIELDS]

    for col in INCLUDED_FIELDS:
        result[col] = result[col].astype("string").str.strip()

    result = result.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "null": pd.NA,
        "NULL": pd.NA,
    })

    if "object" in result.columns:
        result["object"] = result["object"].str.lower()

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


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate created-leads rows."""
    if "team_id" in df.columns and "id" in df.columns:
        return df.drop_duplicates(subset=["team_id", "id"]).reset_index(drop=True)

    if "id" in df.columns:
        return df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    return df.drop_duplicates().reset_index(drop=True)


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple DataFrames into one merged DataFrame."""
    if not dfs:
        raise ValueError("No valid files found to merge.")

    return pd.concat(dfs, ignore_index=True)


def transform_merged_created_leads(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Apply the full transformation pipeline for merged created-leads data."""
    merged = merge_dataframes(dfs)
    merged = deduplicate_rows(merged)
    merged = merged[OUTPUT_FIELDS]
    return merged


# ============================================================
# SOURCE FOLDER CLEANUP
# ============================================================

SOURCE_PREFIX_TO_EMPTY = "activity-created-leads/"


def empty_s3_prefix(s3, bucket: str, prefix: str) -> None:
    """Delete only .csv and .csv.gz files under an S3 prefix."""
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
    empty_s3_prefix(s3, BUCKET, SOURCE_PREFIX_TO_EMPTY)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Main execution flow."""
    s3 = get_s3_client()

    files = list_source_files(
        s3=s3,
        bucket=BUCKET,
        prefix=SOURCE_PREFIX,
        output_key=OUTPUT_KEY,
    )

    if not files:
        print("No files found—done.")
        return

    dfs: list[pd.DataFrame] = []

    for key in files:
        print(f"Reading {key}")
        df = read_csv_from_s3(s3, BUCKET, key)

        if df is None:
            print(f"Skipping empty or invalid CSV: {key}")
            continue

        df = standardize_columns(df)
        df = flatten_recipient_fields(df)
        df = enforce_schema(df, source_key=key)

        if not df.empty:
            dfs.append(df)
        else:
            print(f"Skipped {key} — no valid rows after row-level schema filtering")

    if not dfs:
        print("No valid files found—done.")
        return

    merged = transform_merged_created_leads(dfs)

    if merged.empty:
        print("Merged output has 0 valid rows. Nothing will be written.")
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
