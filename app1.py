
 
 
# import streamlit as st
# import pandas as pd
# import os
# import re
# import json
# import asyncio
# import threading
# import time
 
# try:
#     import nest_asyncio
#     nest_asyncio.apply()
# except ImportError:
#     pass
 
# from serpapi_news import run_serpapi_research
 
# # Lazy import — avoids sniffio/httpx crash on Render startup
# # google.genai triggers httpx → sniffio at import time, which fails
# # in Render's async context before any event loop is running.
# def _get_async_runner():
#     from Mail_claude import _async_email_runner
#     return _async_email_runner
 
# st.set_page_config(page_title="SerpAPI Email Generator", layout="centered")
 
# # ==============================================================================
# # PATHS
# # ==============================================================================
# BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
# RESEARCH_FOLDER    = os.path.join(BASE_DIR, "research_cache")
# EMAIL_CACHE_FOLDER = os.path.join(BASE_DIR, "email_cache")
# OUTPUT_FOLDER      = os.path.join(BASE_DIR, "local_output_files")
 
# _pipeline_lock = threading.Lock()
 
# # ==============================================================================
# # SESSION STATE — same keys as original
# # ==============================================================================
# if "final_csv_data"    not in st.session_state: st.session_state.final_csv_data    = None
# if "final_df_preview"  not in st.session_state: st.session_state.final_df_preview  = None
# if "service_choice"    not in st.session_state: st.session_state.service_choice    = None
# if "pipeline_running"  not in st.session_state: st.session_state.pipeline_running  = False
# if "pipeline_error"    not in st.session_state: st.session_state.pipeline_error    = None
# if "results_store_ref" not in st.session_state: st.session_state.results_store_ref = None
# if "result_holder_ref" not in st.session_state: st.session_state.result_holder_ref = None
# if "uploaded_df"       not in st.session_state: st.session_state.uploaded_df       = None
 
 
# # ==============================================================================
# # HELPERS
# # ==============================================================================
# def inject_first_name(row):
#     """Safely injects the First Name into the Email Body."""
#     body  = str(row.get("Email_Body", ""))
#     fname = str(row.get("First Name", "")).strip()
#     if not fname or fname.lower() in ["nan", "none"]:
#         return body
#     updated_body = re.sub(r"^Hi\s*,", f"Hi {fname},", body, count=1)
#     if updated_body == body and not body.lower().startswith("hi"):
#         updated_body = f"Hi {fname},\n\n{body}"
#     return updated_body
 
# def _safe_get(row, col, default="N/A"):
#     val = row.get(col, default)
#     try:
#         if pd.isna(val): return default
#     except Exception: pass
#     return val
 
 
# # ==============================================================================
# # EMAIL CALLBACK — runs in EMAIL EXECUTOR thread
# # NEVER touch st.session_state here — wrong thread, will crash.
# # Writes only to results_store (plain dict passed by reference).
# # ==============================================================================
# def _make_email_callback(df: pd.DataFrame, service_focus: str, results_store: dict):
 
#     def callback(company_list: list) -> None:
#         if not company_list:
#             return
 
#         os.makedirs(RESEARCH_FOLDER,    exist_ok=True)
#         os.makedirs(EMAIL_CACHE_FOLDER, exist_ok=True)
 
#         mini_rows = []
#         for company_data in company_list:
#             company_name = company_data.get("company", "").strip()
#             if not company_name:
#                 continue
 
#             matched = df[df["Company Name"].astype(str).str.strip().str.lower() == company_name.lower()]
#             if matched.empty:
#                 industry, annual_revenue, total_funding = "Technology", "N/A", "N/A"
#             else:
#                 r              = matched.iloc[0]
#                 industry       = str(_safe_get(r, "Industry", "Technology")).strip()
#                 annual_revenue = _safe_get(r, "Annual Revenue")
#                 total_funding  = _safe_get(r, "Total Funding")
 
#             safe_name = "".join(c for c in company_name if c.isalnum() or c in "._- ").strip().replace(" ", "_").lower()
#             with open(os.path.join(RESEARCH_FOLDER, f"{safe_name}.json"), "w", encoding="utf-8") as f:
#                 json.dump({"company": company_name, "pain_points": company_data.get("pain_points", []), "recent_news": company_data.get("recent_news", [])}, f, indent=4)
 
#             mini_rows.append({"Company Name": company_name, "Industry": industry, "Annual Revenue": annual_revenue, "Total Funding": total_funding})
 
#         if not mini_rows:
#             return
 
#         mini_df = pd.DataFrame(mini_rows)
 
#         _async_email_runner = _get_async_runner()
#         try:
#             batch_result_df = asyncio.run(
#                 _async_email_runner(df=mini_df, json_data_folder=RESEARCH_FOLDER, service_focus=service_focus, email_cache_folder=EMAIL_CACHE_FOLDER)
#             )
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             try:
#                 batch_result_df = loop.run_until_complete(
#                     _async_email_runner(df=mini_df, json_data_folder=RESEARCH_FOLDER, service_focus=service_focus, email_cache_folder=EMAIL_CACHE_FOLDER)
#                 )
#             finally:
#                 loop.close()
 
#         # Write to plain dict only — NOT st.session_state
#         with _pipeline_lock:
#             for _, result_row in batch_result_df.iterrows():
#                 key = str(result_row.get("Company Name", "")).strip().lower()
#                 if key:
#                     results_store[key] = {
#                         "Email_subject": result_row.get("Generated_Email_Subject", ""),
#                         "Email_Body":    result_row.get("Generated_Email_Body",    ""),
#                         "AI_Source":     result_row.get("AI_Source",               ""),
#                     }
 
#     return callback
 
 
# # ==============================================================================
# # PIPELINE RUNNER — background daemon thread
# # Never touches st.session_state — signals via result_holder plain dict.
# # ==============================================================================
# def _run_full_pipeline(df: pd.DataFrame, service_choice: str, results_store: dict, result_holder: dict) -> None:
#     try:
#         run_serpapi_research(df=df, email_callback=_make_email_callback(df, service_choice, results_store), batch_size=10)
 
#         final_df = df.copy()
#         final_df["Email_subject"] = ""
#         final_df["Email_Body"]    = ""
#         final_df["AI_Source"]     = ""
 
#         for idx, row in final_df.iterrows():
#             key = str(row.get("Company Name", "")).strip().lower()
#             if key in results_store:
#                 final_df.at[idx, "Email_subject"] = results_store[key].get("Email_subject", "")
#                 final_df.at[idx, "Email_Body"]    = results_store[key].get("Email_Body",    "")
#                 final_df.at[idx, "AI_Source"]     = results_store[key].get("AI_Source",     "")
 
#         requested_columns = ["First Name", "Last Name", "Company Name", "Email", "Industry", "Email_subject", "Email_Body"]
#         for col in requested_columns:
#             if col not in final_df.columns:
#                 final_df[col] = ""
 
#         filtered_df = final_df[requested_columns].copy()
#         filtered_df["Email_Body"] = filtered_df.apply(inject_first_name, axis=1)
 
#         os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#         filtered_df.to_csv(os.path.join(OUTPUT_FOLDER, f"Final_SerpAPI_Leads_{service_choice}.csv"), index=False, encoding="utf-8-sig")
 
#         result_holder["done"] = filtered_df   # signals main thread: pipeline complete
 
#     except Exception as e:
#         result_holder["error"] = str(e)       # signals main thread: pipeline failed
 
 
# # ==============================================================================
# # MAIN APP — same UI as original, steps stay visible during run
# # ==============================================================================
# def main():
 
#     st.title("✉️ SerpAPI + AI Email Engine")
#     st.markdown("Automate research via Google AI Mode and generate highly personalized outbound emails.")
 
#     # Error banner
#     if st.session_state.pipeline_error:
#         st.error(f"❌ An error occurred: {st.session_state.pipeline_error}")
 
#     # ── ALWAYS VISIBLE: Upload + Service selection ────────────────────────────
#     # These stay on screen the entire time — same as original app.
#     # Only the Run button hides once pipeline starts.
#     st.markdown("### 📥 Step 1: Upload Company Data")
#     uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
 
#     # Store uploaded df in session so it survives reruns
#     if uploaded_file is not None and st.session_state.uploaded_df is None:
#         df_temp = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
#         df_temp.columns = [str(c).strip() for c in df_temp.columns]
#         if "Company Name" not in df_temp.columns:
#             st.error("❌ The uploaded file MUST contain a 'Company Name' column.")
#             st.stop()
#         st.session_state.uploaded_df = df_temp
#         st.success(f"✅ File loaded — {len(df_temp)} rows, {df_temp['Company Name'].nunique()} unique companies.")
 
#     st.markdown("### 🎯 Step 2: Select Service Pitch")
#     service_options = ["AI", "Salesforce", "Combined"]
#     # Keep current selection across reruns
#     current_idx = 0
#     if st.session_state.service_choice in ["ai", "salesforce", "combined"]:
#         current_idx = ["ai", "salesforce", "combined"].index(st.session_state.service_choice)
#     service_choice = st.radio("What service are you pitching?", service_options, index=current_idx, horizontal=True).lower()
#     st.session_state.service_choice = service_choice
 
#     # ── Run button — shown only when ready and not already running/done ───────
#     if (st.session_state.uploaded_df is not None
#             and not st.session_state.pipeline_running
#             and st.session_state.final_csv_data is None):
 
#         if st.button("🚀 Run Search & Generate Emails", type="primary"):
 
#             results_store = {}
#             result_holder = {}
 
#             st.session_state.results_store_ref = results_store
#             st.session_state.result_holder_ref = result_holder
#             st.session_state.pipeline_running  = True
#             st.session_state.pipeline_error    = None
 
#             threading.Thread(
#                 target = _run_full_pipeline,
#                 args   = (st.session_state.uploaded_df, service_choice, results_store, result_holder),
#                 daemon = True,
#             ).start()
#             st.rerun()
 
#     # ── PHASE 2: Running — spinner + live email counter ───────────────────────
#     if st.session_state.pipeline_running:
 
#         result_holder = st.session_state.result_holder_ref
#         results_store = st.session_state.results_store_ref
 
#         if "error" in result_holder:
#             st.session_state.pipeline_error   = result_holder["error"]
#             st.session_state.pipeline_running = False
#             st.rerun()
 
#         elif "done" in result_holder:
#             filtered_df = result_holder["done"]
#             st.session_state.final_df_preview = filtered_df.copy()
#             st.session_state.final_csv_data   = filtered_df.to_csv(index=False).encode("utf-8-sig")
#             st.session_state.pipeline_running = False
#             st.rerun()
 
#         else:
#             with st.spinner("Running SerpAPI research and generating AI emails. This may take a few minutes..."):
#                 emails_so_far = len(results_store)
#                 if emails_so_far:
#                     st.success(f"📧 Emails generated so far: **{emails_so_far}**")
#                 time.sleep(10)
#                 st.rerun()
 
#     # ── PHASE 3: Done — persists until browser refresh ────────────────────────
#     if st.session_state.final_csv_data is not None:
 
#         st.divider()
#         st.markdown("### 🔥 Preview Generated Content")
 
#         # Warning — blank emails
#         blank_mask = (
#             st.session_state.final_df_preview["Email_subject"].isna() |
#             (st.session_state.final_df_preview["Email_subject"].astype(str).str.strip() == "")
#         )
#         failed_companies = st.session_state.final_df_preview[blank_mask]["Company Name"].unique().tolist()
#         if failed_companies:
#             st.warning(
#                 f"⚠️ **{len(failed_companies)} company email(s) could not be generated:** "
#                 f"{', '.join(failed_companies)}\n\n"
#                 f"These rows are included in the CSV with blank email fields."
#             )
 
#         st.dataframe(st.session_state.final_df_preview.head(), width="stretch")
 
#         st.download_button(
#             label     = "📥 Download Final Output (CSV)",
#             data      = st.session_state.final_csv_data,
#             file_name = f"Final_SerpAPI_Leads_{st.session_state.service_choice}.csv",
#             mime      = "text/csv",
#             type      = "primary",
#         )
 
 
# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import os
import re
import json
import asyncio
import threading
import time
 
from serpapi_news import run_serpapi_research
 
# Lazy import — avoids sniffio/httpx crash on Render startup
# google.genai triggers httpx → sniffio at import time, which fails
# in Render's async context before any event loop is running.
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
# SESSION STATE — same keys as original
# ==============================================================================
if "final_csv_data"    not in st.session_state: st.session_state.final_csv_data    = None
if "final_df_preview"  not in st.session_state: st.session_state.final_df_preview  = None
if "service_choice"    not in st.session_state: st.session_state.service_choice    = None
if "pipeline_running"  not in st.session_state: st.session_state.pipeline_running  = False
if "pipeline_error"    not in st.session_state: st.session_state.pipeline_error    = None
if "results_store_ref" not in st.session_state: st.session_state.results_store_ref = None
if "result_holder_ref" not in st.session_state: st.session_state.result_holder_ref = None
if "uploaded_df"       not in st.session_state: st.session_state.uploaded_df       = None
 
 
# ==============================================================================
# HELPERS
# ==============================================================================
def inject_first_name(row):
    """Safely injects the First Name into the Email Body."""
    body  = str(row.get("Email_Body", ""))
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
# EMAIL CALLBACK — runs in EMAIL EXECUTOR thread
# NEVER touch st.session_state here — wrong thread, will crash.
# Writes only to results_store (plain dict passed by reference).
# ==============================================================================
def _make_email_callback(df: pd.DataFrame, service_focus: str, results_store: dict):
 
    def callback(company_list: list) -> None:
        if not company_list:
            return
 
        os.makedirs(RESEARCH_FOLDER,    exist_ok=True)
        os.makedirs(EMAIL_CACHE_FOLDER, exist_ok=True)
 
        mini_rows = []
        for company_data in company_list:
            company_name = company_data.get("company", "").strip()
            if not company_name:
                continue
 
            matched = df[df["Company Name"].astype(str).str.strip().str.lower() == company_name.lower()]
            if matched.empty:
                industry, annual_revenue, total_funding = "Technology", "N/A", "N/A"
            else:
                r              = matched.iloc[0]
                industry       = str(_safe_get(r, "Industry", "Technology")).strip()
                annual_revenue = _safe_get(r, "Annual Revenue")
                total_funding  = _safe_get(r, "Total Funding")
 
            safe_name = "".join(c for c in company_name if c.isalnum() or c in "._- ").strip().replace(" ", "_").lower()
            with open(os.path.join(RESEARCH_FOLDER, f"{safe_name}.json"), "w", encoding="utf-8") as f:
                json.dump({"company": company_name, "pain_points": company_data.get("pain_points", []), "recent_news": company_data.get("recent_news", [])}, f, indent=4)
 
            mini_rows.append({"Company Name": company_name, "Industry": industry, "Annual Revenue": annual_revenue, "Total Funding": total_funding})
 
        if not mini_rows:
            return
 
        mini_df = pd.DataFrame(mini_rows)
 
        # Use run_email_pipeline() which internally manages a persistent per-thread loop.
        # This ensures KeyWorker._lock is always created on the same loop it runs on,
        # preventing 'Timeout should be used inside a task' on Python 3.14.
        from Mail_claude import run_email_pipeline
        batch_result_df = run_email_pipeline(
            df=mini_df,
            json_data_folder=RESEARCH_FOLDER,
            service_focus=service_focus,
            email_cache_folder=EMAIL_CACHE_FOLDER,
        )
 
        # Write to plain dict only — NOT st.session_state
        with _pipeline_lock:
            for _, result_row in batch_result_df.iterrows():
                key = str(result_row.get("Company Name", "")).strip().lower()
                if key:
                    results_store[key] = {
                        "Email_subject": result_row.get("Generated_Email_Subject", ""),
                        "Email_Body":    result_row.get("Generated_Email_Body",    ""),
                        "AI_Source":     result_row.get("AI_Source",               ""),
                    }
 
    return callback
 
 
# ==============================================================================
# PIPELINE RUNNER — background daemon thread
# Never touches st.session_state — signals via result_holder plain dict.
# ==============================================================================
def _run_full_pipeline(df: pd.DataFrame, service_choice: str, results_store: dict, result_holder: dict) -> None:
    try:
        run_serpapi_research(df=df, email_callback=_make_email_callback(df, service_choice, results_store), batch_size=10)
 
        final_df = df.copy()
        final_df["Email_subject"] = ""
        final_df["Email_Body"]    = ""
        final_df["AI_Source"]     = ""
 
        for idx, row in final_df.iterrows():
            key = str(row.get("Company Name", "")).strip().lower()
            if key in results_store:
                final_df.at[idx, "Email_subject"] = results_store[key].get("Email_subject", "")
                final_df.at[idx, "Email_Body"]    = results_store[key].get("Email_Body",    "")
                final_df.at[idx, "AI_Source"]     = results_store[key].get("AI_Source",     "")
 
        requested_columns = ["First Name", "Last Name", "Company Name", "Email", "Industry", "Email_subject", "Email_Body"]
        for col in requested_columns:
            if col not in final_df.columns:
                final_df[col] = ""
 
        filtered_df = final_df[requested_columns].copy()
        filtered_df["Email_Body"] = filtered_df.apply(inject_first_name, axis=1)
 
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filtered_df.to_csv(os.path.join(OUTPUT_FOLDER, f"Final_SerpAPI_Leads_{service_choice}.csv"), index=False, encoding="utf-8-sig")
 
        result_holder["done"] = filtered_df   # signals main thread: pipeline complete
 
    except Exception as e:
        result_holder["error"] = str(e)       # signals main thread: pipeline failed
 
 
# ==============================================================================
# MAIN APP — same UI as original, steps stay visible during run
# ==============================================================================
def main():
 
    st.title("✉️ SerpAPI + AI Email Engine")
    st.markdown("Automate research via Google AI Mode and generate highly personalized outbound emails.")
 
    # Error banner
    if st.session_state.pipeline_error:
        st.error(f"❌ An error occurred: {st.session_state.pipeline_error}")
 
    # ── ALWAYS VISIBLE: Upload + Service selection ────────────────────────────
    # These stay on screen the entire time — same as original app.
    # Only the Run button hides once pipeline starts.
    st.markdown("### 📥 Step 1: Upload Company Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
 
    # Store uploaded df in session so it survives reruns
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
    # Keep current selection across reruns
    current_idx = 0
    if st.session_state.service_choice in ["ai", "salesforce", "combined"]:
        current_idx = ["ai", "salesforce", "combined"].index(st.session_state.service_choice)
    service_choice = st.radio("What service are you pitching?", service_options, index=current_idx, horizontal=True).lower()
    st.session_state.service_choice = service_choice
 
    # ── Run button — shown only when ready and not already running/done ───────
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
                target = _run_full_pipeline,
                args   = (st.session_state.uploaded_df, service_choice, results_store, result_holder),
                daemon = True,
            ).start()
            st.rerun()
 
    # ── PHASE 2: Running — spinner + live email counter ───────────────────────
    if st.session_state.pipeline_running:
 
        result_holder = st.session_state.result_holder_ref
        results_store = st.session_state.results_store_ref
 
        if "error" in result_holder:
            st.session_state.pipeline_error   = result_holder["error"]
            st.session_state.pipeline_running = False
            st.rerun()
 
        elif "done" in result_holder:
            filtered_df = result_holder["done"]
            st.session_state.final_df_preview = filtered_df.copy()
            st.session_state.final_csv_data   = filtered_df.to_csv(index=False).encode("utf-8-sig")
            st.session_state.pipeline_running = False
            st.rerun()
 
        else:
            with st.spinner("Running SerpAPI research and generating AI emails. This may take a few minutes..."):
                emails_so_far = len(results_store)
                if emails_so_far:
                    st.success(f"📧 Emails generated so far: **{emails_so_far}**")
                time.sleep(10)
                st.rerun()
 
    # ── PHASE 3: Done — persists until browser refresh ────────────────────────
    if st.session_state.final_csv_data is not None:
 
        st.divider()
        st.markdown("### 🔥 Preview Generated Content")
 
        # Warning — blank emails
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
 
        st.dataframe(st.session_state.final_df_preview.head(), width="stretch")
 
        st.download_button(
            label     = "📥 Download Final Output (CSV)",
            data      = st.session_state.final_csv_data,
            file_name = f"Final_SerpAPI_Leads_{st.session_state.service_choice}.csv",
            mime      = "text/csv",
            type      = "primary",
        )
 
 
if __name__ == "__main__":
    main()
