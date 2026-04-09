from __future__ import annotations

import base64
import concurrent.futures as cf
import csv
import gzip
import io
import json
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# CONFIG
# =========================================================
SECRETS_NAME = "mailshake-api"
BUCKET = "mailshake-analysis"

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
PER_PAGE = 100

# Lower default throttle. Only slow down hard when API actually rate-limits us.
API_CALL_DELAY_SECONDS = 0.05
FAILURE_PAUSE_SECONDS = 2

# Process multiple teams at once.
MAX_WORKERS = 6

# Recycle session occasionally
SESSION_RESET_EVERY_N_PAGES = 250

# Optional: compress uploads to reduce S3 upload time / storage
COMPRESS_UPLOAD = True

TEXT_COLUMN_TOKENS = ("body", "html", "rawbody", "plaintextbody")

CORRUPTED_PATTERNS = [
    r"ARC-Message-Signature:.*",
    r"ARC-Authentication-Results:.*",
    r"ARC-Seal:.*",
    r"X-MS-Exchange-AntiSpam-MessageData.*",
    r"DKIM-Signature:.*",
    r"Authentication-Results:.*",
    r"\bb=[A-Za-z0-9+/=]{100,}\b",
]

ENDPOINTS = {
    "replies": {
        "url": "https://api.mailshake.com/2017-04-01/activity/replies",
        "prefix": "activity-replies/",
        "extra_params": {"excludeBody": "true"},
        "clean_text_fields": True,
    },
    "sent": {
        "url": "https://api.mailshake.com/2017-04-01/activity/sent",
        "prefix": "activity-sent/",
        "extra_params": {},
        "clean_text_fields": True,
    },
    "opens": {
        "url": "https://api.mailshake.com/2017-04-01/activity/opens",
        "prefix": "activity-opens/",
        "extra_params": {},
        "clean_text_fields": False,
    },
    "clicks": {
        "url": "https://api.mailshake.com/2017-04-01/activity/clicks",
        "prefix": "activity-clicks/",
        "extra_params": {},
        "clean_text_fields": False,
    },
    "created_leads": {
        "url": "https://api.mailshake.com/2017-04-01/activity/created-leads",
        "prefix": "activity-created-leads/",
        "extra_params": {},
        "clean_text_fields": False,
    },
}


# =========================================================
# AUTH
# =========================================================
def get_mailshake_keys(secret_name: str) -> dict[str, str]:
    secrets_client = boto3.client("secretsmanager")
    secret = secrets_client.get_secret_value(SecretId=secret_name)
    return json.loads(secret["SecretString"])


def build_headers(api_key: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("utf-8")
    return {
        "Authorization": f"Basic {encoded}",
        "Accept": "application/json",
    }


# =========================================================
# REQUEST
# =========================================================
def parse_mailshake_retry_wait_seconds(response_text: str) -> Optional[int]:
    if not response_text:
        return None

    match = re.search(
        r"try again after:\s*([0-9T:\-\.]+Z)",
        response_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    retry_at_raw = match.group(1)

    try:
        retry_at = datetime.fromisoformat(retry_at_raw.replace("Z", "+00:00"))
        now_utc = datetime.now(timezone.utc)
        wait_seconds = int((retry_at - now_utc).total_seconds())
        return max(1, wait_seconds)
    except Exception:
        return None


def request_with_retries(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    params: dict,
    team_id: str,
) -> requests.Response:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return response

            if response.status_code == 429:
                retry_wait = parse_mailshake_retry_wait_seconds(response.text)
                if retry_wait is None:
                    retry_wait = min(60, (2 ** attempt) + random.uniform(0, 1))

                print(
                    f"[team={team_id}] 429 on attempt {attempt}/{MAX_RETRIES}; "
                    f"sleeping {retry_wait:.1f}s"
                )
                time.sleep(retry_wait)
                last_error = RuntimeError(f"429: {response.text}")
                continue

            if 500 <= response.status_code < 600:
                sleep_seconds = min(30, (2 ** attempt) + random.uniform(0, 1))
                print(
                    f"[team={team_id}] {response.status_code} on attempt "
                    f"{attempt}/{MAX_RETRIES}; sleeping {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
                last_error = RuntimeError(
                    f"Retryable error {response.status_code}: {response.text}"
                )
                continue

            raise RuntimeError(
                f"Non-retryable error {response.status_code}: {response.text}"
            )

        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break

            sleep_seconds = min(30, (2 ** attempt) + random.uniform(0, 1))
            print(
                f"[team={team_id}] request exception on attempt "
                f"{attempt}/{MAX_RETRIES}: {exc}; sleeping {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Request failed after retries: {last_error}")


# =========================================================
# CLEANING
# =========================================================
def clean_text(value):
    if pd.isna(value):
        return value

    text = str(value)

    # Cheap skip for non-HTML plain text
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

    for pattern in CORRUPTED_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        col for col in df.columns
        if any(token in col.lower() for token in TEXT_COLUMN_TOKENS)
    ]

    if not text_cols:
        return df

    for col in text_cols:
        df[col] = df[col].map(clean_text)

    return df


# =========================================================
# TRANSFORM
# =========================================================
def normalize(records: list[dict]) -> pd.DataFrame:
    return pd.json_normalize(records, sep=".")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return df


def transform(records: list[dict], team_id: str, clean: bool = False) -> pd.DataFrame:
    df = normalize(records)
    df.insert(0, "team_id", team_id)
    df = standardize_columns(df)

    if clean:
        df = clean_text_columns(df)

    return df


# =========================================================
# FETCH
# =========================================================
def fetch_all_records(
    team_id: str,
    api_key: str,
    url: str,
    extra_params: Optional[dict] = None,
) -> list[dict]:
    session = requests.Session()
    headers = build_headers(api_key)

    all_rows: list[dict] = []
    next_token = None
    extra_params = extra_params or {}
    page_count = 0

    while True:
        if API_CALL_DELAY_SECONDS > 0:
            time.sleep(API_CALL_DELAY_SECONDS)

        params = {"perPage": PER_PAGE, **extra_params}
        if next_token:
            params["nextToken"] = next_token

        response = request_with_retries(session, url, headers, params, team_id)

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse JSON for team={team_id}, url={url}: {exc}"
            ) from exc

        results = data.get("results", [])
        if not results:
            break

        all_rows.extend(results)
        page_count += 1

        if page_count % 25 == 0:
            print(
                f"[team={team_id}] fetched {page_count} pages / "
                f"{len(all_rows)} rows from {url}"
            )

        next_token = data.get("nextToken")
        if not next_token:
            break

        if page_count % SESSION_RESET_EVERY_N_PAGES == 0:
            session.close()
            session = requests.Session()

    session.close()
    return all_rows


# =========================================================
# S3
# =========================================================
def upload(df: pd.DataFrame, bucket: str, key: str) -> None:
    s3 = boto3.client("s3")

    if COMPRESS_UPLOAD:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as gz:
            gz.write(buffer.getvalue().encode("utf-8"))

        s3.put_object(
            Bucket=bucket,
            Key=f"{key}.gz",
            Body=gz_buffer.getvalue(),
            ContentType="text/csv",
            ContentEncoding="gzip",
            ServerSideEncryption="AES256",
        )
    else:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="text/csv",
            ServerSideEncryption="AES256",
        )


# =========================================================
# TEAM JOB
# =========================================================
def process_team_endpoint(
    endpoint_name: str,
    cfg: dict,
    team_id: str,
    api_key: str,
) -> dict:
    started = time.perf_counter()
    print(f"[START] endpoint={endpoint_name} team={team_id}")

    rows = fetch_all_records(
        team_id=team_id,
        api_key=api_key,
        url=cfg["url"],
        extra_params=cfg.get("extra_params"),
    )

    if not rows:
        elapsed = time.perf_counter() - started
        print(f"[DONE] endpoint={endpoint_name} team={team_id} rows=0 elapsed={elapsed:.1f}s")
        return {
            "endpoint": endpoint_name,
            "team_id": team_id,
            "rows": 0,
            "elapsed_seconds": elapsed,
            "status": "no_data",
        }

    df = transform(
        rows,
        team_id=team_id,
        clean=cfg["clean_text_fields"],
    )

    key = f"{cfg['prefix']}team_{team_id}.csv"
    upload(df, BUCKET, key)

    elapsed = time.perf_counter() - started
    print(
        f"[DONE] endpoint={endpoint_name} team={team_id} "
        f"rows={len(df)} elapsed={elapsed:.1f}s"
    )

    return {
        "endpoint": endpoint_name,
        "team_id": team_id,
        "rows": len(df),
        "elapsed_seconds": elapsed,
        "status": "ok",
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    overall_start = time.perf_counter()
    keys = get_mailshake_keys(SECRETS_NAME)

    for endpoint_name, cfg in ENDPOINTS.items():
        endpoint_start = time.perf_counter()
        print(f"\n=== STARTING ENDPOINT: {endpoint_name} ===")

        futures = []
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for team_id, api_key in keys.items():
                futures.append(
                    executor.submit(
                        process_team_endpoint,
                        endpoint_name,
                        cfg,
                        team_id,
                        api_key,
                    )
                )

            for future in cf.as_completed(futures):
                try:
                    result = future.result()
                    print(f"[RESULT] {result}")
                except Exception as e:
                    print(f"[FAILED] endpoint={endpoint_name}: {e}")
                    time.sleep(FAILURE_PAUSE_SECONDS)

        endpoint_elapsed = time.perf_counter() - endpoint_start
        print(
            f"=== FINISHED ENDPOINT: {endpoint_name} "
            f"in {endpoint_elapsed / 60:.2f} min ==="
        )

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\nALL DONE in {overall_elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
