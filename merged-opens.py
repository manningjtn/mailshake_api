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
SOURCE_PREFIX = "activity-opens/"

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_KEY = f"merged/merged-opens/opens_{timestamp}.csv"

INCLUDED_FIELDS = [
    "team_id",
    "object",
    "id",
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
    "recipient_fields_link",
    "recipient_fields_position",
    "recipient_fields_date_applied",
    "recipient_fields_account",
    "recipient_fields_phonenumber",
    "recipient_fields_facebookurl",
    "recipient_fields_instagramid",
    "recipient_fields_linkedinurl",
    "recipient_fields_twitterid",
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


def read_csv_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    """
    Read a CSV or CSV.GZ file from S3 into a pandas DataFrame.
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
# CLEANING / TRANSFORMATION HELPERS
# ============================================================

def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnamed columns typically created by malformed CSV exports.
    """
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]


def safe_parse_dict(value: str) -> dict:
    """
    Safely parse a stringified Python dictionary.
    """
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return {}
    return {}


def flatten_nested_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Flatten one nested object column into prefixed columns.
    """
    if column_name not in df.columns:
        return df

    parsed = df[column_name].apply(safe_parse_dict)
    flat = pd.json_normalize(parsed).add_prefix(f"{column_name}.")
    df = df.drop(columns=[column_name]).reset_index(drop=True)
    flat = flat.reindex(df.index)

    return pd.concat([df, flat], axis=1)


def flatten_possible_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten known nested-object columns if they are present.
    """
    for col in ["recipient", "campaign", "parent"]:
        df = flatten_nested_column(df, col)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase with underscores.
    """
    result = df.copy()
    result.columns = (
        result.columns
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return result


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove temporary or redundant nested columns that should not remain in the
    final merged dataset.
    """
    return df.drop(
        columns=[
            c
            for c in [
                "recipient",
                "campaign",
                "parent",
                "recipient_fields",
                "parent_message",
            ]
            if c in df.columns
        ],
        errors="ignore",
    )


def enforce_schema(df: pd.DataFrame, source_key: str | None = None) -> pd.DataFrame:
    """
    Force DataFrame into the expected schema.

    Behavior:
        - Never skip the whole file just because required columns are missing.
        - Add any missing included fields as null.
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
    """
    Remove duplicate rows and duplicate column names.
    """
    result = df.loc[:, ~df.columns.duplicated()]
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def clean_cell(value):
    """
    Clean a single cell value.
    """
    if isinstance(value, str):
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
    return value


def clean_dataframe_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cell-level cleanup across the full DataFrame.
    """
    return df.apply(lambda col: col.map(clean_cell))


def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames into one merged DataFrame.
    """
    if not dfs:
        raise ValueError("No valid source CSV files found to merge.")

    return pd.concat(dfs, ignore_index=True)


def normalize_mixed_datetime_series(series: pd.Series) -> pd.Series:
    """
    Normalize mixed date formats like:
    - 26-06-2025 00:00
    - 1/7/2025 0:00

    Output:
    - 2025-06-26 00:00:00
    """
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


def transform_merged_opens(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = merge_dataframes(dfs)
    merged = drop_redundant_columns(merged)
    merged = deduplicate_rows(merged)

    for col in merged.columns:
        if "date" in col.lower():
            merged[col] = normalize_mixed_datetime_series(merged[col])

    merged = merged[OUTPUT_FIELDS]
    merged = clean_dataframe_cells(merged)
    return merged


# ============================================================
# SOURCE FOLDER CLEANUP
# ============================================================

SOURCE_PREFIX_TO_EMPTY = "activity-opens/"


def empty_s3_prefix(s3, bucket: str, prefix: str) -> None:
    """
    Delete all objects under an S3 prefix.
    """
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects_to_delete = []

    for page in pages:
        for item in page.get("Contents", []):
            objects_to_delete.append({"Key": item["Key"]})

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

    print(f"Emptied s3://{bucket}/{prefix}")


def empty_source_folder(s3) -> None:
    empty_s3_prefix(s3, BUCKET, SOURCE_PREFIX_TO_EMPTY)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main execution flow.

    Steps:
        - list all source CSV / CSV.GZ files in the opens prefix
        - read each file from S3
        - clean and flatten any remaining nested fields
        - normalize column names
        - enforce schema per file
        - keep only non-empty normalized DataFrames
        - merge all files together
        - deduplicate and enforce output schema
        - write the final merged CSV back to S3
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
        df = drop_unnamed_columns(df)
        df = flatten_possible_nested_columns(df)
        df = standardize_columns(df)
        df = enforce_schema(df, source_key=key)

        if not df.empty:
            dfs.append(df)
        else:
            print(f"Skipped {key} — no valid rows after row-level schema filtering")

    if not dfs:
        print("No valid rows found in any source files. Nothing to merge.")
        return

    merged = transform_merged_opens(dfs)

    if merged.empty:
        print("Merged output has 0 valid open rows. Nothing will be written.")
        return

    write_csv_to_s3(
        s3=s3,
        df=merged,
        bucket=BUCKET,
        key=OUTPUT_KEY,
    )

    print(f"Wrote cleaned file to s3://{BUCKET}/{OUTPUT_KEY}")

    print("Emptying source folder...")
    empty_source_folder(s3)


if __name__ == "__main__":
    main()