


import streamlit as st
import pandas as pd
import os
import re
import json
import asyncio
import threading
import time
import unicodedata

from serpapi_news import run_serpapi_research

def _get_async_runner():
    from Mail_claude import _async_email_runner
    return _async_email_runner

st.set_page_config(page_title="SerpAPI Email Generator", layout="centered")

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
RESEARCH_FOLDER    = os.path.join(BASE_DIR, "research_cache")
EMAIL_CACHE_FOLDER = os.path.join(BASE_DIR, "email_cache")
OUTPUT_FOLDER      = os.path.join(BASE_DIR, "local_output_files")

_pipeline_lock = threading.Lock()

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "final_csv_data"    not in st.session_state: st.session_state.final_csv_data    = None
if "final_df_preview"  not in st.session_state: st.session_state.final_df_preview  = None
if "service_choice"    not in st.session_state: st.session_state.service_choice    = None
if "pipeline_running"  not in st.session_state: st.session_state.pipeline_running  = False
if "pipeline_error"    not in st.session_state: st.session_state.pipeline_error    = None
if "results_store_ref" not in st.session_state: st.session_state.results_store_ref = None
if "result_holder_ref" not in st.session_state: st.session_state.result_holder_ref = None
if "uploaded_df"       not in st.session_state: st.session_state.uploaded_df       = None
if "final_csv_clean"   not in st.session_state: st.session_state.final_csv_clean   = None

# ==============================================================================
# HELPERS
# ==============================================================================

def _normalize_name(name: str) -> str:
    """
    Single normalization used everywhere.
    'Alexza / Ferrer'          -> 'alexza_ferrer'
    'Alexza Pharmaceuticals / Ferrer' -> 'alexza_pharmaceuticals_ferrer'
    'Hawaiʻi Gas'              -> 'hawaii_gas'
    """
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = "".join(c for c in name if c.isalnum() or c in " ")
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"_+", "_", name)
    return name


def _fuzzy_match(key: str, store: dict):
    """
    Find best matching key in store.
    First tries exact match, then checks if one is substring of other.
    'alexza_ferrer' will match 'alexza_pharmaceuticals_ferrer'
    """
    if key in store:
        return key
    key_parts = set(key.split("_"))
    best = None
    best_score = 0
    for rkey in store:
        rkey_parts = set(rkey.split("_"))
        common = len(key_parts & rkey_parts)
        if common > best_score:
            best_score = common
            best = rkey
    # Only match if at least half the parts overlap
    if best and best_score >= max(1, len(key_parts) // 2):
        return best
    return None


def inject_first_name(row):
    """Safely injects the First Name into the Email Body."""
    body = str(row.get("Email_Body", ""))
    if not body.strip():
        return ""
    fname = str(row.get("First Name", "")).strip()
    if not fname or fname.lower() in ["nan", "none"]:
        return body
    updated_body = re.sub(r"^Hi\s*,", f"Hi {fname},", body, count=1)
    if updated_body == body and not body.lower().startswith("hi"):
        updated_body = f"Hi {fname},\n\n{body}"
    return updated_body


def _safe_get(row, col, default="N/A"):
    val = row.get(col, default)
    try:
        if pd.isna(val): return default
    except Exception: pass
    return val


# ==============================================================================
# EMAIL CALLBACK
# ==============================================================================
def _make_email_callback(df: pd.DataFrame, service_focus: str, results_store: dict):

    def callback(company_list: list) -> None:
        if not company_list:
            return

        os.makedirs(RESEARCH_FOLDER,    exist_ok=True)
        os.makedirs(EMAIL_CACHE_FOLDER, exist_ok=True)

        # csv_name_map: normalized_key -> original CSV company name
        # This is built BEFORE Mail_claude runs
        csv_name_map = {}
        mini_rows = []

        for company_data in company_list:
            company_name = company_data.get("company", "").strip()
            if not company_name:
                continue

            matched = df[df["Company Name"].astype(str).str.strip().str.lower() == company_name.lower()]
            if matched.empty:
                # Try partial match against original CSV
                for _, dfrow in df.iterrows():
                    csv_co = str(dfrow.get("Company Name", "")).strip()
                    if _normalize_name(csv_co) in _normalize_name(company_name) or \
                       _normalize_name(company_name) in _normalize_name(csv_co):
                        matched = df[df["Company Name"] == csv_co]
                        break

            if matched.empty:
                industry, annual_revenue, total_funding = "Technology", "N/A", "N/A"
                original_csv_name = company_name
            else:
                r              = matched.iloc[0]
                industry       = str(_safe_get(r, "Industry", "Technology")).strip()
                annual_revenue = _safe_get(r, "Annual Revenue")
                total_funding  = _safe_get(r, "Total Funding")
                original_csv_name = str(r.get("Company Name", company_name)).strip()

            # Save research JSON using normalized name
            safe_name = _normalize_name(company_name)
            with open(os.path.join(RESEARCH_FOLDER, f"{safe_name}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "company":     company_name,
                    "pain_points": company_data.get("pain_points", []),
                    "recent_news": company_data.get("recent_news", [])
                }, f, indent=4)

            # Map: normalized SerpAPI name -> original CSV name
            csv_name_map[safe_name] = original_csv_name
            # Also map normalized CSV name -> original CSV name
            csv_name_map[_normalize_name(original_csv_name)] = original_csv_name

            mini_rows.append({
                "Company Name":  company_name,
                "Industry":      industry,
                "Annual Revenue": annual_revenue,
                "Total Funding":  total_funding,
            })

        if not mini_rows:
            return

        mini_df = pd.DataFrame(mini_rows)

        from Mail_claude import run_email_pipeline
        batch_result_df = run_email_pipeline(
            df=mini_df,
            json_data_folder=RESEARCH_FOLDER,
            service_focus=service_focus,
            email_cache_folder=EMAIL_CACHE_FOLDER,
        )

        with _pipeline_lock:
            for _, result_row in batch_result_df.iterrows():
                result_name     = str(result_row.get("Company Name", "")).strip()
                result_norm     = _normalize_name(result_name)

                # Step 1: direct match
                # Step 2: fuzzy match via shared word parts
                store_key = _fuzzy_match(result_norm, {_normalize_name(k): k for k in csv_name_map.values()})

                # Use original CSV name as the results_store key (lowercased)
                if store_key:
                    final_key = csv_name_map.get(store_key, csv_name_map.get(result_norm, result_norm))
                else:
                    final_key = result_name

                # Normalize final key for consistent lookup
                lookup_key = _normalize_name(final_key)

                results_store[lookup_key] = {
                    "Email_subject": result_row.get("Generated_Email_Subject", ""),
                    "Email_Body":    result_row.get("Generated_Email_Body",    ""),
                    "AI_Source":     result_row.get("AI_Source",               ""),
                }

    return callback


# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================
def _run_full_pipeline(df: pd.DataFrame, service_choice: str, results_store: dict, result_holder: dict) -> None:
    try:
        run_serpapi_research(
            df=df,
            email_callback=_make_email_callback(df, service_choice, results_store),
            batch_size=10
        )

        final_df = df.copy()
        final_df["Email_subject"] = ""
        final_df["Email_Body"]    = ""
        final_df["AI_Source"]     = ""

        for idx, row in final_df.iterrows():
            csv_name = str(row.get("Company Name", "")).strip()
            norm_key = _normalize_name(csv_name)

            matched_key = _fuzzy_match(norm_key, results_store)

            if matched_key:
                final_df.at[idx, "Email_subject"] = results_store[matched_key].get("Email_subject", "")
                final_df.at[idx, "Email_Body"]    = results_store[matched_key].get("Email_Body",    "")
                final_df.at[idx, "AI_Source"]     = results_store[matched_key].get("AI_Source",     "")

        requested_columns = ["First Name", "Last Name", "Company Name", "Email", "Industry", "Email_subject", "Email_Body"]
        for col in requested_columns:
            if col not in final_df.columns:
                final_df[col] = ""

        filtered_df = final_df[requested_columns].copy()
        filtered_df["Email_Body"] = filtered_df.apply(inject_first_name, axis=1)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filtered_df.to_csv(
            os.path.join(OUTPUT_FOLDER, f"Final_SerpAPI_Leads_{service_choice}.csv"),
            index=False, encoding="utf-8-sig"
        )

        clean_df = filtered_df[
            filtered_df["Email_subject"].notna() &
            (filtered_df["Email_subject"].astype(str).str.strip() != "")
        ].copy()

        result_holder["done"]       = filtered_df
        result_holder["done_clean"] = clean_df

    except Exception as e:
        import traceback
        result_holder["error"] = str(e) + "\n" + traceback.format_exc()


# ==============================================================================
# MAIN APP
# ==============================================================================
def main():

    st.title("✉️ SerpAPI + AI Email Engine")
    st.markdown("Automate research via Google AI Mode and generate highly personalized outbound emails.")

    if st.session_state.pipeline_error:
        st.error(f"❌ An error occurred: {st.session_state.pipeline_error}")

    st.markdown("### 📥 Step 1: Upload Company Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None and st.session_state.uploaded_df is None:
        df_temp = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df_temp.columns = [str(c).strip() for c in df_temp.columns]
        if "Company Name" not in df_temp.columns:
            st.error("❌ The uploaded file MUST contain a 'Company Name' column.")
            st.stop()
        st.session_state.uploaded_df = df_temp
        st.success(f"✅ File loaded — {len(df_temp)} rows, {df_temp['Company Name'].nunique()} unique companies.")

    st.markdown("### 🎯 Step 2: Select Service Pitch")
    service_options = ["AI", "Salesforce", "Combined"]
    current_idx = 0
    if st.session_state.service_choice in ["ai", "salesforce", "combined"]:
        current_idx = ["ai", "salesforce", "combined"].index(st.session_state.service_choice)
    service_choice = st.radio("What service are you pitching?", service_options, index=current_idx, horizontal=True).lower()
    st.session_state.service_choice = service_choice

    if (st.session_state.uploaded_df is not None
            and not st.session_state.pipeline_running
            and st.session_state.final_csv_data is None):

        if st.button("🚀 Run Search & Generate Emails", type="primary"):
            results_store = {}
            result_holder = {}
            st.session_state.results_store_ref = results_store
            st.session_state.result_holder_ref = result_holder
            st.session_state.pipeline_running  = True
            st.session_state.pipeline_error    = None

            threading.Thread(
                target=_run_full_pipeline,
                args=(st.session_state.uploaded_df, service_choice, results_store, result_holder),
                daemon=True,
            ).start()
            st.rerun()

    if st.session_state.pipeline_running:
        result_holder = st.session_state.result_holder_ref
        results_store = st.session_state.results_store_ref

        if "error" in result_holder:
            st.session_state.pipeline_error   = result_holder["error"]
            st.session_state.pipeline_running = False
            st.rerun()

        elif "done" in result_holder:
            filtered_df = result_holder["done"]
            clean_df    = result_holder.get("done_clean", filtered_df)
            st.session_state.final_df_preview = filtered_df.copy()
            st.session_state.final_csv_data   = filtered_df.to_csv(index=False).encode("utf-8-sig")
            st.session_state.final_csv_clean  = clean_df.to_csv(index=False).encode("utf-8-sig")
            st.session_state.pipeline_running = False
            st.rerun()

        else:
            with st.spinner("Running SerpAPI research and generating AI emails. This may take a few minutes..."):
                emails_so_far = len(results_store)
                if emails_so_far:
                    st.success(f"📧 Emails generated so far: **{emails_so_far}**")
                time.sleep(10)
                st.rerun()

    if st.session_state.final_csv_data is not None:
        st.divider()
        st.markdown("### 🔥 Preview Generated Content")

        blank_mask = (
            st.session_state.final_df_preview["Email_subject"].isna() |
            (st.session_state.final_df_preview["Email_subject"].astype(str).str.strip() == "")
        )
        failed_companies = st.session_state.final_df_preview[blank_mask]["Company Name"].unique().tolist()
        if failed_companies:
            st.warning(
                f"⚠️ **{len(failed_companies)} company email(s) could not be generated:** "
                f"{', '.join(failed_companies)}\n\n"
                f"These rows are included in the CSV with blank email fields."
            )

        # st.dataframe(st.session_state.final_df_preview, use_container_width=True)
        st.dataframe(st.session_state.final_df_preview, width="stretch")

        st.download_button(
            label     = "📥 Download Final Output (CSV)",
            data      = st.session_state.final_csv_data,
            file_name = f"Final_SerpAPI_Leads_{st.session_state.service_choice}.csv",
            mime      = "text/csv",
            type      = "primary",
        )

        if failed_companies and st.session_state.final_csv_clean is not None:
            st.download_button(
                label     = f"✅ Download Clean Output — {len(failed_companies)} blank row(s) removed (CSV)",
                data      = st.session_state.final_csv_clean,
                file_name = f"Final_SerpAPI_Leads_{st.session_state.service_choice}_clean.csv",
                mime      = "text/csv",
            )


if __name__ == "__main__":
    main()



