# import pandas as pd
# import json
# import time
# from serpapi import GoogleSearch
 
# SERPAPI_KEY = "b3464f9406301ef4d4a501ae8d08b66dd06e38ca33f4e8c73c3f465b936ebb0f"
# CSV_FILE = r"C:\Users\user\Desktop\Client Finder AI\new_for_deployement\Reverse.csv"
# OUTPUT_FILE = "ai_mode_research.json"
# import json
# import os
 
# OUTPUT_FOLDER = "structured_company_data"
 
# if not os.path.exists(OUTPUT_FOLDER):
#     os.makedirs(OUTPUT_FOLDER)
 
# def extract_companies_from_response(response):
#     text_blocks = response.get("text_blocks", [])
 
#     full_json_string = ""
 
#     for block in text_blocks:
#         if block.get("type") == "code_block":
#             full_json_string += block.get("code", "")
 
#     full_json_string = full_json_string.strip()
 
#     if full_json_string.startswith("[") and full_json_string.endswith("]"):
#         try:
#             company_data_list = json.loads(full_json_string)
 
#             for company_data in company_data_list:
#                 save_company_json(company_data)
 
#         except json.JSONDecodeError:
#             print("⚠ JSON parsing failed — likely truncated response.")
#     else:
#         print("⚠ No valid JSON block found.")
 
 
# def save_company_json(company_data):
#     company_name = company_data.get("company", "unknown")
 
#     safe_filename = (
#         "".join(c for c in company_name if c.isalnum() or c in "._- ")
#         .strip()
#         .replace(" ", "_")
#         .lower()
#     )
 
#     file_path = os.path.join(OUTPUT_FOLDER, f"{safe_filename}.json")
 
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(company_data, f, indent=4, ensure_ascii=False)
 
#     print(f"Saved: {company_name}")
 
 
# def fetch_ai_mode(companies_batch):
#     company_text = ", ".join(companies_batch)
 
#     prompt = f"""
# Act as a structured business research engine.
 
# For each company listed:
 
# Return STRICT JSON only.
# Do NOT add commentary.
# Keep each company under 120 words total.
 
# Format:
 
# [
#   {{
#     "company": "",
#     "pain_points": ["point1", "point2"],
#     "recent_news": [
#         {{
#             "title": "",
#             "source": ""
#         }}
#     ]
#   }}
# ]
 
# Companies:
# {company_text}
# """
 
 
#     params = {
#         "engine": "google_ai_mode",
#         "q": prompt,
#         "api_key": SERPAPI_KEY,
#         "hl": "en",
#         "gl": "us"
#     }
 
#     search = GoogleSearch(params)
#     results = search.get_dict()
 
#     return results
 
 
# def main():
#     df = pd.read_csv(CSV_FILE)
 
#     # Clean + deduplicate
#     df["clean_name"] = df["Company Name"].str.strip().str.lower()
#     unique_companies = (
#         df.dropna(subset=["clean_name"])
#           .drop_duplicates(subset=["clean_name"])
#     )
 
#     company_list = unique_companies["Company Name"].tolist()
 
#     batch_size = 10  # Keep small for stability
#     final_output = []
 
#     for i in range(0, len(company_list), batch_size):
#         batch = company_list[i:i+batch_size]
#         print(f"Processing batch {i} to {i+len(batch)}")
 
#         try:
#             response = fetch_ai_mode(batch)
#             extract_companies_from_response(response)
#             text_blocks = response.get("text_blocks", [])
#             references = response.get("references", [])
 
#             final_output.append({
#                 "companies_requested": batch,
#                 "ai_text_blocks": text_blocks,
#                 "references": references
#             })
 
#             time.sleep(2)
 
#         except Exception as e:
#             print(f"Error: {e}")
 
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         json.dump(final_output, f, indent=4)
 
#     print("Done.")
 
 
# if __name__ == "__main__":
#     main()
 

import os
import json
import time
import pandas as pd
from serpapi import GoogleSearch
from api_rotating_claude import get_serpapi_key

def get_safe_filename(company_name: str) -> str:
    return "".join(c for c in company_name if c.isalnum() or c in "._- ").strip().replace(" ", "_").lower()

def save_company_json(company_data: dict, output_folder: str):
    company_name = company_data.get("company", "unknown")
    safe_filename = get_safe_filename(company_name)
    file_path = os.path.join(output_folder, f"{safe_filename}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(company_data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved AI Research: {company_name}")

# def extract_companies_from_response(response: dict, output_folder: str):
#     text_blocks = response.get("text_blocks", [])
#     full_json_string = ""
#     for block in text_blocks:
#         if block.get("type") == "code_block":
#             full_json_string += block.get("code", "")
            
#     full_json_string = full_json_string.strip()
#     if full_json_string.startswith("[") and full_json_string.endswith("]"):
#         try:
#             company_data_list = json.loads(full_json_string)
#             for company_data in company_data_list:
#                 save_company_json(company_data, output_folder)
#         except json.JSONDecodeError:
#             print("⚠ JSON parsing failed — response likely truncated.")
#     else:
#         print("⚠ No valid JSON block found in response.")

import re  # 🟢 ADD THIS at the very top of your file

def extract_companies_from_response(response: dict, output_folder: str):
    """
    Scans the entire AI response for a JSON array [...] regardless of 
    formatting (paragraph vs code_block) and saves it.
    """
    text_blocks = response.get("text_blocks", [])
    full_text = ""
    
    # 1. Combine all snippets and code blocks into one big string
    for block in text_blocks:
        full_text += block.get("snippet", "") + block.get("code", "")
            
    # 2. Use Regex to find the first '[' and the last ']'
    # re.DOTALL ensures it captures multiple lines
    match = re.search(r'\[.*\]', full_text, re.DOTALL)
    
    if match:
        json_string = match.group(0).strip()
        try:
            # 3. Parse the matched string into a Python list
            company_data_list = json.loads(json_string)
            for company_data in company_data_list:
                save_company_json(company_data, output_folder)
        except json.JSONDecodeError as e:
            print(f"⚠ Extraction found text that looks like JSON, but it's malformed: {e}")
    else:
        print("⚠ No valid JSON array [...] found anywhere in the AI's response text.")

def fetch_ai_mode(companies_batch: list) -> dict:
    company_text = ", ".join(companies_batch)
    prompt = f"""
Act as a structured business research engine.
For each company listed:
Return STRICT JSON only. Do NOT add commentary. Keep each company under 120 words total.
Format:
[
  {{
    "company": "",
    "pain_points": ["point1", "point2"],
    "recent_news": [
        {{
            "title": "",
            "source": ""
        }}
    ]
  }}
]
Companies:
{company_text}
"""
    params = {
        "engine": "google_ai_mode",
        "q": prompt,
        "api_key": get_serpapi_key(delay=1.0),
        "hl": "en",
        "gl": "us"
    }
    search = GoogleSearch(params)
    return search.get_dict()


def run_serpapi_research(df, output_folder="structured_company_data", batch_size=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clean and deduplicate
    df_clean = df.copy()
    df_clean["clean_name"] = df_clean["Company Name"].astype(str).str.strip()
    unique_companies = df_clean.dropna(subset=["clean_name"]).drop_duplicates(subset=["clean_name"])
    all_companies = unique_companies["clean_name"].tolist()

    # --- CACHING: Only process companies we haven't researched yet ---
    companies_to_process = []
    for company in all_companies:
        safe_name = get_safe_filename(company)
        if not os.path.exists(os.path.join(output_folder, f"{safe_name}.json")):
            companies_to_process.append(company)
        else:
            print(f"⏩ Skipping {company}: Research already exists.")

    if not companies_to_process:
        print("✅ All companies already researched.")
        return True

    # Process in batches
    for i in range(0, len(companies_to_process), batch_size):
        batch = companies_to_process[i:i + batch_size]
        print(f"🔍 Processing batch {i} to {i + len(batch)}")
        try:
            response = fetch_ai_mode(batch)
            extract_companies_from_response(response, output_folder)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            
    return True

# ---------------------------------------------------------
# NEW FUNCTION: For testing/running a single company
# ---------------------------------------------------------
def run_single_company_research(company_name: str, output_folder="structured_company_data"):
    """Fetches research for a single company without needing a CSV file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    safe_name = get_safe_filename(company_name)
    file_path = os.path.join(output_folder, f"{safe_name}.json")

    # Caching check
    if os.path.exists(file_path):
        print(f"⏩ Skipping {company_name}: Research already exists at {file_path}")
        return True

    print(f"🔍 Processing single company: {company_name}")
    try:
        # Pass the single company as a list containing one item
        response = fetch_ai_mode([company_name])
        
        # 🟢 ADD THIS: Print the raw response beautifully to debug
        print("\n--- RAW API RESPONSE ---")
        print(json.dumps(response, indent=4))
        print("------------------------\n")

        extract_companies_from_response(response, output_folder)
    except Exception as e:
        print(f"❌ Error processing {company_name}: {e}")

if __name__ == "__main__":

    # --- OPTION 1: Run for a single company (Great for testing) ---
    target_company = "AnavClouds Software Solutions"
    run_single_company_research(target_company)

    # CSV_FILE = r"C:\Users\user\Desktop\Client Finder AI\new_for_deployement\Reverse.csv"
    # try:
    #     test_df = pd.read_csv(CSV_FILE).head(15)
    #     run_serpapi_research(test_df)
    # except Exception as e:
    #     print(f"Error: {e}")