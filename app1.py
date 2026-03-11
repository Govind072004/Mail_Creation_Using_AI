import streamlit as st
import pandas as pd
import os
import re

# Import the new SerpAPI modular functions
from serpapi_news import run_serpapi_research
from Mail_claude import run_serpapi_email_generation

st.set_page_config(page_title="SerpAPI Email Generator", layout="centered")

def inject_first_name(row):
    """Safely injects the First Name into the Email Body."""
    body = str(row.get('Email_Body', ''))
    fname = str(row.get('First Name', '')).strip()
    if not fname or fname.lower() in ['nan', 'none']:
        return body
    updated_body = re.sub(r'^Hi\s*,', f'Hi {fname},', body, count=1)
    if updated_body == body and not body.lower().startswith("hi"):
        updated_body = f"Hi {fname},\n\n{body}"
    return updated_body

def main():
    st.title("✉️ SerpAPI + AI Email Engine")
    st.markdown("Automate research via Google AI Mode and generate highly personalized outbound emails.")

    if "final_csv_data" not in st.session_state:
        st.session_state.final_csv_data = None
        st.session_state.final_df_preview = None

    st.markdown("### 📥 Step 1: Upload Company Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    st.markdown("### 🎯 Step 2: Select Service Pitch")
    service_choice = st.radio("What service are you pitching?", ["AI", "Salesforce", "Combined"], horizontal=True)
    service_choice = service_choice.lower()

    if uploaded_file:
        if st.button("🚀 Run Search & Generate Emails", type="primary"):
            try:
                # Read Data
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                df.columns = [str(c).strip() for c in df.columns]
                
                if "Company Name" not in df.columns:
                    st.error("❌ The uploaded file MUST contain a 'Company Name' column.")
                    st.stop()

                progress_text = st.empty()
                progress_bar = st.progress(0)

                # --- Phase 1: SerpAPI Research ---
                progress_text.text("🔍 Phase 1: Running SerpAPI Deep Research...")
                progress_bar.progress(20)
                
                # output_folder = "structured_company_data"
                output_folder = "/tmp/structured_company_data"
                run_serpapi_research(df, output_folder=output_folder, batch_size=10)
                
                # --- Phase 2: Email Generation ---
                progress_text.text(f"✍️ Phase 2: Generating Personalized Emails (Focus: {service_choice})...")
                progress_bar.progress(60)
                
                final_email_df = run_serpapi_email_generation(df, json_data_folder=output_folder, service_focus=service_choice,email_cache_folder="/tmp/email_generation_cache_serpapi",)
                
                # --- Phase 3: Format Columns & Inject Name ---
                progress_text.text("📊 Formatting Output File and Injecting Names...")
                progress_bar.progress(90)
                
                final_email_df.rename(columns={
                    "Generated_Email_Subject": "Email_subject", 
                    "Generated_Email_Body": "Email_Body"
                }, inplace=True)

                requested_columns = ['First Name', 'Last Name', 'Company Name', 'Email', 'Industry', 'Email_subject', 'Email_Body']
                for col in requested_columns:
                    if col not in final_email_df.columns:
                        final_email_df[col] = ""  

                filtered_output_df = final_email_df[requested_columns].copy()
                filtered_output_df['Email_Body'] = filtered_output_df.apply(inject_first_name, axis=1)

                # --- Phase 4: AUTOMATICALLY SAVE TO LOCAL FOLDER ---
                # local_save_dir = "local_output_files"
                local_save_dir = "/tmp/local_output_files"
                if not os.path.exists(local_save_dir):
                    os.makedirs(local_save_dir)
                
                local_filename = os.path.join(local_save_dir, f"Final_SerpAPI_Leads_{service_choice}.csv")
                # filtered_output_df.to_csv(local_filename, index=False, encoding="utf-8")
                filtered_output_df.to_csv(local_filename, index=False, encoding="utf-8-sig")

                progress_bar.progress(100)
                progress_text.text("✅ Process Complete!")
                st.success("Emails successfully generated!")
                
                st.session_state.final_df_preview = filtered_output_df.copy()
                # st.session_state.final_csv_data = filtered_output_df.to_csv(index=False).encode("utf-8")
                st.session_state.final_csv_data = filtered_output_df.to_csv(index=False).encode("utf-8-sig")
                st.session_state.service_choice = service_choice
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

    if st.session_state.final_csv_data is not None:
        st.divider()
        st.markdown("### 🔥 Preview Generated Content")
        # st.dataframe(st.session_state.final_df_preview.head(), use_container_width=True)
        st.dataframe(st.session_state.final_df_preview.head(), width="stretch")
        st.download_button(
            label="📥 Download Final Output (CSV)",
            data=st.session_state.final_csv_data,
            file_name=f"Final_SerpAPI_Leads_{st.session_state.service_choice}.csv",
            mime="text/csv",
            type="primary"
        )

if __name__ == "__main__":
    main()
