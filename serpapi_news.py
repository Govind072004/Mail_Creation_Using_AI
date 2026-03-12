
import os
import re
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from serpapi import GoogleSearch
from dotenv import load_dotenv

from api_rotating_claude import get_serpapi_key

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)


def _step_log(step: str, message: str, emoji: str = "▶"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  {step:<8}  {emoji}  {message}")


CACHE_FOLDER = os.path.join(BASE_DIR, "research_cache")


def _cache_path(company_name: str) -> str:
    safe = "".join(c for c in company_name if c.isalnum() or c in "._- ").strip().replace(" ", "_").lower()
    return os.path.join(CACHE_FOLDER, f"{safe}.json")


def load_local_cache() -> set:
    _step_log("STEP 2", "Loading cached names from local research_cache folder...")
    try:
        if not os.path.exists(CACHE_FOLDER):
            _step_log("STEP 2", "Cache folder not found — starting fresh.", "⚠️")
            return set()
        cached = set()
        for fname in os.listdir(CACHE_FOLDER):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(CACHE_FOLDER, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("company", "").strip().lower()
                    if name:
                        cached.add(name)
                except Exception:
                    pass
        _step_log("STEP 2", f"Local cache loaded: {len(cached)} companies already researched.", "✅")
        return cached
    except Exception as e:
        _step_log("STEP 2", f"Cache read error — continuing without cache. Error: {e}", "⚠️")
        return set()


def get_company_from_cache(company_name: str) -> dict | None:
    path = _cache_path(company_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_company_to_cache(company_data: dict) -> None:
    name = company_data.get("company", "").strip()
    if not name:
        return
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    path = _cache_path(name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(company_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        _step_log("CACHE", f"Failed to save cache for {name}: {e}", "⚠️")


def _repair_json(raw_text: str) -> list:
    if not raw_text or not raw_text.strip():
        return []

    cleaned = raw_text.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json
        repaired = repair_json(cleaned)
        result   = json.loads(repaired)
        if isinstance(result, list):
            _step_log("PARSE", "JSON repaired successfully using json-repair library.", "🔧")
            return result
    except (ImportError, Exception):
        pass

    _step_log("PARSE", f"All JSON repair attempts failed. Raw text preview: {cleaned[:200]}...", "❌")
    return []


def _parse_serpapi_response(response: dict) -> list:
    text_blocks = response.get("text_blocks", [])
    full_text   = ""
    for block in text_blocks:
        full_text += block.get("snippet", "") + block.get("code", "")

    match = re.search(r'\[.*\]', full_text, re.DOTALL)
    if match:
        return _repair_json(match.group(0).strip())

    _step_log("PARSE", "No JSON array [...] found anywhere in SerpAPI response.", "⚠️")
    return []


def _fetch_one_batch(batch: list, batch_label: str) -> list:
    company_text = ", ".join(batch)
    _step_log("STEP 3", f"{batch_label} → Fetching {len(batch)} companies via SerpAPI...", "🔍")

    prompt = f"""
Act as a structured business research engine.
For each company listed below, return STRICT JSON only.
Do NOT add any commentary, markdown, preamble, or explanation.
Keep each company entry under 120 words total.

Return format — a JSON array, nothing else:
[
  {{
    "company": "Exact Company Name",
    "pain_points": [
      "Specific business pain point 1",
      "Specific business pain point 2",
      "Specific business pain point 3"
    ],
    "recent_news": [
      {{"title": "News headline", "source": "Source name"}},
      {{"title": "News headline", "source": "Source name"}}
    ]
  }}
]

Companies to research:
{company_text}
"""

    params = {
        "engine":  "google_ai_mode",
        "q":       prompt,
        "api_key": get_serpapi_key(),
        "hl":      "en",
        "gl":      "us",
    }

    try:
        response     = GoogleSearch(params).get_dict()
        company_list = _parse_serpapi_response(response)
        _step_log("STEP 3", f"{batch_label} → Returned {len(company_list)}/{len(batch)} companies.", "✅")
        return company_list
    except Exception as e:
        _step_log("STEP 3", f"{batch_label} → SerpAPI fetch failed: {e}", "❌")
        return []


def _hand_to_email_pipeline(company_list: list, batch_label: str, email_callback) -> None:
    if not company_list:
        return
    if email_callback is None:
        _step_log("EMAIL", f"{batch_label} → No email callback set — research-only mode.", "⚠️")
        return
    _step_log("EMAIL", f"{batch_label} → Handing {len(company_list)} companies to Email Creation.", "📧")
    try:
        email_callback(company_list)
        _step_log("EMAIL", f"{batch_label} → Email Creation handoff complete.", "✅")
    except Exception as e:
        _step_log("EMAIL", f"{batch_label} → Email Creation callback raised error: {e}", "❌")


def run_serpapi_research(
    df,
    email_callback            = None,
    output_folder: str        = "structured_company_data",
    batch_size: int           = 10,
    max_parallel_fetches: int = 2,
    max_email_workers: int    = 4,
) -> dict:

    _step_log("STEP 1", "Extracting unique company names from uploaded sheet...")

    raw_names     = df["Company Name"].astype(str).str.strip().dropna().tolist()
    unique_set    = {name for name in raw_names if name and name.lower() != "nan"}
    all_companies = sorted(unique_set)

    _step_log("STEP 1", f"Found {len(raw_names)} rows → {len(all_companies)} unique after deduplication.", "✅")

    cached_names = load_local_cache()

    companies_to_fetch  = []
    cached_company_data = []
    skipped_count       = 0

    for company in all_companies:
        if company.strip().lower() in cached_names:
            skipped_count += 1
            data = get_company_from_cache(company)
            if data:
                cached_company_data.append(data)
        else:
            companies_to_fetch.append(company)

    _step_log(
        "STEP 1",
        f"Pipeline summary → Total: {len(all_companies)} | "
        f"Already cached (skip SerpAPI): {skipped_count} | "
        f"To fetch now: {len(companies_to_fetch)}",
        "📊"
    )

    session_data = {}

    if email_callback and cached_company_data:
        _step_log("STEP 2", f"Sending {len(cached_company_data)} cached companies to email pipeline...")
        with ThreadPoolExecutor(max_workers=max_email_workers) as cached_email_executor:
            for i in range(0, len(cached_company_data), batch_size):
                batch = cached_company_data[i : i + batch_size]
                cached_email_executor.submit(
                    _hand_to_email_pipeline,
                    batch,
                    f"Cache-Batch {i // batch_size + 1}",
                    email_callback,
                )
        for d in cached_company_data:
            name = d.get("company", "").strip()
            if name:
                session_data[name.lower()] = {
                    "pain_points": d.get("pain_points", []),
                    "recent_news": d.get("recent_news",  []),
                }

    if not companies_to_fetch:
        _step_log("STEP 1", "All companies already cached. Zero SerpAPI credits used.", "✅")
        return session_data

    total_batches = -(-len(companies_to_fetch) // batch_size)

    _step_log("STEP 3", f"Starting parallel pipeline — {len(companies_to_fetch)} companies | {total_batches} batches.")
    _step_log("STEP 3", f"FETCH POOL: {max_parallel_fetches} threads | EMAIL POOL: {max_email_workers} threads (separate).")

    with ThreadPoolExecutor(max_workers=max_parallel_fetches) as fetch_executor, \
         ThreadPoolExecutor(max_workers=max_email_workers)    as email_executor:

        fetch_futures = {}
        for i in range(0, len(companies_to_fetch), batch_size):
            batch       = companies_to_fetch[i : i + batch_size]
            batch_num   = i // batch_size + 1
            batch_label = f"Batch {batch_num}/{total_batches}"
            future = fetch_executor.submit(_fetch_one_batch, batch, batch_label)
            fetch_futures[future] = batch_label

        for future in as_completed(fetch_futures):
            batch_label  = fetch_futures[future]
            company_list = future.result()

            if not company_list:
                _step_log("STEP 3", f"{batch_label} → No data returned. Skipping.", "⚠️")
                continue

            for company_data in company_list:
                save_company_to_cache(company_data)
                name = company_data.get("company", "").strip()
                if name:
                    session_data[name.lower()] = {
                        "pain_points": company_data.get("pain_points", []),
                        "recent_news": company_data.get("recent_news",  []),
                    }

            email_executor.submit(
                _hand_to_email_pipeline,
                company_list,
                batch_label,
                email_callback,
            )

        _step_log("EMAIL", "Waiting for all email creation tasks to complete...")

    _step_log("DONE", f"Pipeline complete. {len(session_data)} companies processed.", "🎉")
    return session_data


def run_single_company_research(company_name: str, email_callback=None, output_folder: str = "structured_company_data") -> dict | None:
    _step_log("TEST", f"Single company mode: '{company_name}'")

    cached = load_local_cache()
    if company_name.strip().lower() in cached:
        _step_log("TEST", "Already in local cache — no SerpAPI call needed.", "⏩")
        existing = get_company_from_cache(company_name)
        if existing:
            _step_log("TEST", f"Existing data:\n{json.dumps(existing, indent=2)}", "📋")
        return existing

    company_list = _fetch_one_batch([company_name], "Test 1/1")

    if not company_list:
        _step_log("TEST", "No data returned from SerpAPI.", "❌")
        return None

    for data in company_list:
        save_company_to_cache(data)
        _step_log("TEST", f"Fetched & cached:\n{json.dumps(data, indent=2)}", "✅")

    if email_callback:
        _hand_to_email_pipeline(company_list, "Test 1/1", email_callback)

    return company_list[0] if company_list else None


if __name__ == "__main__":
    target_company = "AnavClouds Software Solutions"
    run_single_company_research(target_company)
