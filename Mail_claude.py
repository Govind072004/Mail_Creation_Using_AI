
 
 
 
import os
import json
import pandas as pd
import asyncio
import re
import sys
import logging
import time
 
from google import genai
from google.genai import types
from groq import AsyncGroq
from cerebras.cloud.sdk import AsyncCerebras
 
from openai import AsyncAzureOpenAI
from api_rotating_claude import (
    KeyWorker,      build_worker_pool,
    get_azure_config,
)
 
# ==============================================================================
# LOGGING SETUP
# ==============================================================================
 
# Logging: stdout only — no FileHandler.
# Render filesystem is ephemeral; log files would be lost on restart.
# All logs visible in Render dashboard and local terminal both.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
 
# Mute noisy third-party loggers
for _noisy in [
    "google", "google.genai", "google.generativeai",
    "httpx", "google_genai.models", "google_genai.types",
    "asyncio", "urllib3", "httpcore",
]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
 
 
# ==============================================================================
# GLOBAL CIRCUIT BREAKER
# ==============================================================================
 
CONSECUTIVE_FAILURES  = 0
MAX_FAILURES          = 7
CIRCUIT_BREAKER_TRIPPED = False
 
_cb_lock: asyncio.Lock | None = None
 
def _get_cb_lock() -> asyncio.Lock:
    global _cb_lock
    if _cb_lock is None:
        _cb_lock = asyncio.Lock()
    return _cb_lock
 
 
# ==============================================================================
# API CALL FUNCTIONS
# ==============================================================================
 
# ==============================================================================
# SYSTEM PROMPT — fixed persona, injected via system role on every call
# Keeps user prompt focused on company data only → better quality per token
# ==============================================================================
 
SYSTEM_PROMPT = """You are a senior B2B sales copywriter at AnavClouds with 12 years writing cold outbound for enterprise tech companies. You've written thousands of emails. You know what gets replies and what goes to spam.
 
Your writing style:
- You write like a busy, sharp professional — short sentences, real observations, zero fluff
- You never write marketing copy. You write peer-to-peer business notes.
- You use contractions naturally (don't, we're, it's, they've)
- Your sentences are uneven in length — that's intentional
- You never start with "I wanted to" and never end with a question or CTA
- You notice one specific thing about the company and react to it — not summarize it
 
Your output discipline:
- You follow the exact format given — no extra sections, no sign-offs
- You stop writing immediately after the 4th bullet
- You never use banned words even once — if you catch yourself, you rewrite
- You never produce symmetric bullets — each one feels different in length and style"""
 
 
async def call_gemini_async(prompt: str, api_key: str) -> str:
    """
    Google Gemini 2.5 Flash — PRIMARY
    System instruction used for persona, user prompt for company data.
    """
    client   = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text
 
 
async def call_cerebras_async(prompt: str, api_key: str) -> str:
    """
    Cerebras gpt-oss-120b — SECONDARY
    System role used for persona, user role for company task.
    """
    client   = AsyncCerebras(api_key=api_key)
    response = await client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.25,
        max_completion_tokens=1500,
    )
    return response.choices[0].message.content
 
 
async def call_groq_async(prompt: str, api_key: str) -> str:
    """
    Groq llama-4-scout-17b — OVERFLOW
    System role used for persona, user role for company task.
    """
    client   = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.25,
    )
    return response.choices[0].message.content
 
async def call_azure_async(prompt: str) -> str:
    """
    Azure OpenAI GPT-4o Mini — EMERGENCY FALLBACK
    """
    config = get_azure_config()
 
    client = AsyncAzureOpenAI(
        api_key        = config["api_key"],
        azure_endpoint = config["endpoint"],
        api_version    = config["api_version"],
    )
 
    response = await client.chat.completions.create(
        model       = config["deployment"],
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature = 0.25,
        max_tokens  = 1500,
    )
    return response.choices[0].message.content
 
# ==============================================================================
# SERVICE CAPABILITY BLOCKS — only relevant one injected per prompt
# ==============================================================================
 
_SERVICE_BLOCK_AI = """
* AI agents built on custom LLMs and RAG — answers from internal data, not generic responses
* Predictive models for lead scoring, demand forecasting, churn — so the team works the right accounts
* ETL pipelines and data warehouses built for real-time analytics, not quarterly reports
* End-to-end process automation with Python and AI frameworks — repetitive work removed, not reassigned
* Specialists in AI, ML, data engineering — screened, tested, contract or remote, onboarding in days"""
 
_SERVICE_BLOCK_SALESFORCE = """
* Sales, Service, Health, and Financial Services Cloud implementations built around actual workflows — not default configs
* Custom Apex, Lightning Web Components, and Flow automations that cut manual steps teams have lived with for years
* AppExchange integrations, Pardot/Marketing Cloud campaigns, multi-org migrations — handled without business disruption
* 24/7 managed services with proactive health checks — technical debt caught before it becomes an incident
* Salesforce developers, admins, consultants across clouds — contract or remote, onboarding in days"""
 
_SERVICE_BLOCK_COMBINED = """
* Predictive AI/ML models wired into Salesforce dashboards — lead scoring and forecasting that actually reflects reality
* Workflows combining Lightning Components, Flows, and Python-based AI — CRM and analytics moving together, not separately
* AI agents querying Salesforce data in real time — anomalies caught, insights surfaced without someone pulling reports
* Specialists skilled in both Salesforce and AI — no handoff gaps, onboarding in days, contract or remote"""
 
_SERVICE_BLOCKS = {
    "ai":         _SERVICE_BLOCK_AI,
    "salesforce": _SERVICE_BLOCK_SALESFORCE,
    "combined":   _SERVICE_BLOCK_COMBINED,
}
 
# ==============================================================================
# PROMPT BUILDER
# ==============================================================================
 
def _build_email_prompt(
    company:    str,
    industry:   str,
    financials: str,
    market_news:str,
    pain_points:str,
    service_focus: str,
) -> str:
    """Optimized prompt — system prompt handles persona, user prompt handles task."""
 
    capabilities = _SERVICE_BLOCKS.get(service_focus.lower(), _SERVICE_BLOCK_AI)
    

    return f"""
You are a senior B2B sales copywriter for AnavClouds. Write a hyper-personalized outbound email that feels human and manually written — never templated or marketing-heavy.

SELL: {service_focus} only. Mention "AnavClouds" once, in Block 2 only.

---
COMPANY DATA
- Company: {company}
- Industry: {industry}
- Financials: {financials}
- Market News: {market_news}
- Pain Points: {pain_points}
---

THINK BEFORE WRITING (internal only — do not output):
1. Extract ONE strong signal from market_news or financials (Growth / Operational / Tech / GTM).
2. Pick the 2 strongest pains from pain_points. Convert each to an outcome phrase.
3. Map those pains to capabilities below. Frame as outcomes, not features. Tone: curious peer, not vendor.

CAPABILITIES TO USE:
{capabilities}

---
WRITING RULES — ENFORCE ALL:

Tone: Conversational, human, busy professional. Use contractions. Vary sentence length. No brochure copy.

FORBIDDEN PHRASES: reach out, touch base, circle back, game-changer, cutting-edge, best-in-class, world-class, I wanted to connect, Hope this finds you well, Let me know if you're interested, Would love to, Excited to share, Scale your business, Drive results, Unlock potential, Quick call, Hop on a call, Free consultation, Revolutionize, Transform, Disrupt, Just checking in

BANNED WORDS: accelerate, certified, optimize, enhance, leverage, synergy, streamline, empower, solutions, deliverables, bandwidth, mission-critical, investment, fast, new, Here

NO exclamation marks. NO all-caps. NO CTA. NO sign-off. NO ending question. Email ends after final bullet.

Subject format: [Desired Outcome] without [Core Friction] — no tools/services/buzzwords mentioned.

---
EMAIL STRUCTURE:

SUBJECT:
[Outcome without Friction — peer-note tone]

Hi ,

[Block 1 — 2 sentences MAX:
Sentence 1: Reference exactly ONE specific news item or financial signal.
Start with "I noticed" or "I saw" — react like you read it this morning,
don't summarize or review it. One sharp observation only.
Sentence 2: Connect to a natural business direction. No pain mention.
No industry name. No generic sector statements.

RULES:
- Do NOT summarize the news. React to it.
- NO "imagine". NO generic sector statements. NO industry name.
- Must feel like you read about them this morning.]

[Block 2 — 2 sentences only:
Line 1: ALWAYS start with "At AnavClouds," — then describe what we do as the logical next layer for where this company is heading. Mention 2-3 work areas naturally in prose. Never bullet here.
Line 2: "We've helped teams [outcome of pain 1] and [outcome of pain 2]." — mapped directly to THIS company's pain points, not generic.]

[Pick ONE transition randomly — end with colon:
"Here are some ways we can help:"
"Here's what usually helps in situations like this :"
"A few practical ways teams simplify this :"
"What tends to work well in cases like this :"
"Here's what teams often find useful :"]

• Bullet 1 — [Direct fix for strongest pain — outcome framed, conversational]

• Bullet 2 — [Broader {industry} workflow, data setup, or tech debt improvement]

• Bullet 3 — [Direct fix for second strongest pain — framed as outcome, not staffing]

• Bullet 4 —  [specialist-level technical depth only a {service_focus} expert can deliver for {industry} — name the exact method or architecture, never a generic capability, never RAG as default, never staffing or hiring]

BULLET RULES: blank line after transition colon, blank line between each bullet, use only •, no symmetry, no marketing copy.

FINAL CHECK: No banned word? Block 2 starts with "At AnavClouds,"? Bullet 4 is technical not staffing? No CTA? Ends after last bullet? → Output.
"""

#     return f"""Write one outbound email for this company. Follow every rule below exactly.
 
# COMPANY
# - Name: {company}
# - Industry: {industry}
# - Financials: {financials}
# - Recent News: {market_news}
# - Pain Points: {pain_points}
# - Pitch: {service_focus} only
 
# ---
# INTERNAL REASONING — do this silently, output nothing from this section:
 
# Step 1 — Signal: Find ONE concrete signal in the news or financials (a specific number, product launch, expansion, restructure, funding event). Not a vague trend. One real thing.
 
# Step 2 — Pains: From the pain_points list, pick the 2 that would cost this company the most if left unfixed. Convert each to a short outcome phrase (what good looks like, not what's broken).
 
# Step 3 — Opener test: Draft the opening line. Ask — does it sound like you read about them this morning, or like you researched them? Rewrite until it's the former.
 
# Step 4 — Bullet check: After writing bullets, read them aloud. If any two sound like the same length or structure, rewrite one. Asymmetry is the goal.
 
# ---
# CAPABILITIES (use only these, framed as outcomes):
# {capabilities}
 
# ---
# HARD RULES:
 
# FORBIDDEN PHRASES — if any appear, rewrite that sentence:
# reach out, touch base, circle back, game-changer, cutting-edge, best-in-class, world-class, I wanted to connect, Hope this finds you well, Let me know if you're interested, Would love to, Excited to share, Scale your business, Drive results, Unlock potential, Quick call, Hop on a call, Free consultation, Revolutionize, Transform, Disrupt, Just checking in, I came across
 
# BANNED WORDS — not even once:
# accelerate, certified, optimize, enhance, leverage, synergy, streamline, empower, solutions, deliverables, bandwidth, mission-critical, investment, fast, new
 
# NO exclamation marks. NO all-caps. NO CTA. NO sign-off. NO ending question.
# Email stops immediately after bullet 4. Nothing after it.
 
# ---
# OUTPUT FORMAT — follow this exactly, no deviations:
 
# SUBJECT:
# [One line. Format: specific outcome + "without" + real friction. No tools, no buzzwords, no company name.]
 
# Hi ,
 
# [Opening — 2 sentences only.
# Sentence 1: Name the ONE specific signal you found. React to it like a peer — don't explain it, don't summarize it. Start with something other than "I noticed" or "I saw".
# Sentence 2: Connect that signal to a natural business direction. No pain mention yet. No generic industry statements.]
 
# [Positioning — 2 sentences only.
# Sentence 1: Introduce AnavClouds once, naturally — what we do as the logical next step for where they're heading. Do NOT use "At AnavClouds" as the opener. Vary the entry.
# Sentence 2: "We've helped teams [outcome of pain 1] and [outcome of pain 2]." — specific to THIS company's pains, not generic.]
 
# [One transition line randomly chosen from: "Here's what usually helps in situations like this :" / "A few practical ways teams simplify this :" / "What tends to work well in cases like this :" / "Here's what teams often find useful :"]
 
# * [Bullet 1 — direct fix for strongest pain. Outcome-framed. Conversational, not polished.]
 
# * [Bullet 2 — broader {industry} workflow, data, or infrastructure improvement. Different length from bullet 1.]
 
# * [Bullet 3 — fix for second pain. Framed as result, not as a service being offered.]
 
# * [Bullet 4 — one specific {service_focus} capability that requires real specialist depth — tied directly to {industry}. Should feel technical and specific, not generic.]
# """
 
 
 
# ==============================================================================
# WORKER COROUTINE
# ==============================================================================
 
async def _email_worker_loop(
    worker_id:      int,
    key_worker:     KeyWorker,
    queue:          asyncio.Queue,
    results:        dict,
    total_expected: int,
    email_cache_folder: str,
    service_focus:  str,
    worker_pool:    list,
) -> None:
    
    global CONSECUTIVE_FAILURES, CIRCUIT_BREAKER_TRIPPED
    provider_label = key_worker.provider.capitalize()
 
    while True:
        if CIRCUIT_BREAKER_TRIPPED:
            break
 
        if len(results) >= total_expected:
            break
 
        try:
            task = await asyncio.wait_for(queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            if len(results) >= total_expected:
                break
            continue
 
        company     = task["company"]
        index       = task["index"]
        full_prompt = task["prompt"]
        cache_path  = task["cache_path"]
        retry_count = task.get("retry_count", 0)
 
        if retry_count >= 3:
            # Azure yahan nahi — _retry_failed_emails ka Azure pool handle karega
            logging.warning(f"⚠️  [W{worker_id:02d}|{provider_label}] {company} — Max retries reached. Marking Failed for Azure fallback.")
            results[index] = {
                "company":    company,
                "source":     "Failed",
                "raw_email":  "ERROR: Max retries reached — queued for Azure fallback",
                "cache_path": cache_path,
                "prompt":     full_prompt,
            }
            queue.task_done()
            continue
 
        ready = await key_worker.wait_and_acquire()
        if not ready:
            # BUG FIX 1: Was break — worker permanently exited even when tasks remained.
            # Now requeue the task and continue so worker loops back.
            # Workers only exit when all results done or circuit breaker trips.
            logging.warning(
                f"⚠️  Worker {worker_id} ({provider_label}) not ready. Requeueing {company}."
            )
            task["retry_count"] = retry_count
            await queue.put(task)
            queue.task_done()
            await asyncio.sleep(2.0)
            continue
 
        logging.info(f"[W{worker_id:02d}|{provider_label}] → {company} (attempt {retry_count + 1})")
 
        try:
            if key_worker.provider == "gemini":
                raw_email = await asyncio.wait_for(call_gemini_async(full_prompt, key_worker.api_key), timeout=35.0)
            elif key_worker.provider == "cerebras":
                raw_email = await asyncio.wait_for(call_cerebras_async(full_prompt, key_worker.api_key), timeout=35.0)
            else:
                raw_email = await asyncio.wait_for(call_groq_async(full_prompt, key_worker.api_key), timeout=35.0)
            
            raw_email = raw_email or "ERROR: API returned empty response"
 
            async with _get_cb_lock():
                CONSECUTIVE_FAILURES = 0
 
            key_worker.reset_retry_count()  
 
            subject_line, email_body = _parse_email_output(raw_email)
            if subject_line and email_body and "ERROR" not in raw_email:
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"subject": subject_line, "body": email_body, "source": provider_label}, f, indent=4)
                logging.info(f"✅ [W{worker_id:02d}|{provider_label}] {company} — Done & Cached.")
            else:
                logging.warning(f"⚠️  [W{worker_id:02d}|{provider_label}] {company} — Parsing Issue.")
 
            results[index] = {
                "company":    company,
                "source":     provider_label,
                "raw_email":  raw_email,
                "cache_path": cache_path,
                "prompt":     full_prompt,
            }
            queue.task_done()
 
        except Exception as exc:
            err_lower = str(exc).lower()
 
            if isinstance(exc, asyncio.TimeoutError) or "timeout" in err_lower:
                logging.warning(f"⚠️  [W{worker_id:02d}|{provider_label}] Timeout on {company}. Requeueing (attempt {retry_count + 1}).")
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()
                continue
 
            elif any(kw in err_lower for kw in ["429", "rate_limit", "rate limit", "quota_exceeded", "resource_exhausted", "too many requests"]):
                key_worker.mark_429()
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()
 
            elif any(kw in err_lower for kw in ["daily", "exceeded your daily", "monthly", "billing"]):
                key_worker.mark_daily_exhausted()
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()
                break
 
            else:
                logging.error(f"❌ [W{worker_id:02d}|{provider_label}] Hard error: {exc}")
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()
 
 
# ==============================================================================
# RESULT PARSER
# ==============================================================================
 
def _parse_email_output(raw_email: str) -> tuple[str, str]:
    if not raw_email:
        return "", "ERROR: API returned empty response"
 
    clean_text = raw_email.strip()
    clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
    clean_text = re.sub(r'\n```$', '', clean_text)
    clean_text = clean_text.strip()
 
    if clean_text.startswith("ERROR"):
        return "", clean_text
 
    subject_line = ""
    email_body = clean_text
    pre_body = ""
 
    body_match = re.search(r'(?m)^((?:Hi|Hello|Hey|Dear)[\s,].*)', clean_text, re.DOTALL | re.IGNORECASE)
 
    if body_match:
        email_body = body_match.group(1).strip()
        pre_body = clean_text[:body_match.start()].strip()
    else:
        parts = re.split(r'\n{2,}', clean_text, maxsplit=1)
        if len(parts) == 2:
            pre_body, email_body = parts[0].strip(), parts[1].strip()
        else:
            pre_body = ""
            email_body = clean_text
 
    if pre_body:
        sub_clean = re.sub(r'(?i)\*?\*?SUBJECT:\*?\*?\s*', '', pre_body).strip()
        sub_clean = re.sub(r'-\s*\n\s*', '', sub_clean)
        sub_clean = re.sub(r'\s*\n\s*', ' ', sub_clean)
        subject_line = re.sub(r'^"|"$', '', sub_clean).strip()
 
    if subject_line and email_body.startswith(subject_line):
        email_body = email_body[len(subject_line):].strip()
 
    email_body = email_body.strip()
    if not email_body:
        return "", "ERROR: Email body is completely empty after parsing."
 
    word_count = len(email_body.split())
    if word_count < 40:
        return "", f"ERROR: Truncated email (Only {word_count} words). Fails word count check."
 
    bullet_matches = re.findall(r'(?m)^[\s]*[\*•\-\–\—]|â€¢', email_body)
    if len(bullet_matches) < 4:
        return "", f"ERROR: Incomplete generation. Found only {len(bullet_matches)} bullets, expected 4."
 
    if email_body[-1] not in ['.', '!', '?', '"', '\'']:
        return "", "ERROR: Email body cut off mid-sentence (No ending punctuation)."
 
    last_line = email_body.split('\n')[-1].strip()
    if len(last_line.split()) < 4 and last_line[-1] not in ['.', '!', '?']:
        return "", f"ERROR: Last bullet point seems cut off ('{last_line}')."
 
    return subject_line, email_body
 
 
async def _retry_failed_emails(
    df_output:          pd.DataFrame,
    original_df:        pd.DataFrame,
    json_data_folder:   str,
    service_focus:      str,
    email_cache_folder: str,
    worker_pool:        list,
) -> pd.DataFrame:
 
    retry_workers = [
        w for w in worker_pool
        if w.provider in ("cerebras", "groq")
    ]
 
    error_mask = (
        df_output["Generated_Email_Body"].astype(str).str.contains("ERROR", na=False) |
        df_output["Generated_Email_Subject"].isna() |
        (df_output["Generated_Email_Subject"].astype(str).str.strip() == "")
    )
    failed_indices = df_output[error_mask].index.tolist()
 
    if not failed_indices:
        logging.info("✅ No failed emails — skipping retry.")
        return df_output
 
    logging.info(f"\n🔁 AUTO-RETRY START — {len(failed_indices)} failed emails (Cerebras+Groq only)\n")
 
    queue   = asyncio.Queue()
    results = {}
 
    for index in failed_indices:
        row          = original_df.loc[index]
        company_name = str(row.get("Company Name", "")).strip()
        industry     = str(row.get("Industry", "Technology"))
        financial_intel = (
            f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
            f"Total Funding: {row.get('Total Funding', 'N/A')}"
        )
 
        safe_filename = (
            "".join(c for c in company_name if c.isalnum() or c in "._- ")
            .strip().replace(" ", "_").lower()
        )
 
        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        pain_points_str = "Not available."
        market_news     = "No recent market updates available."
 
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                research = json.load(f)
            if "pain_points" in research:
                pain_points_str = "\n".join([f"- {p}" for p in research["pain_points"]])
            if "recent_news" in research:
                market_news = "\n---\n".join([
                    f"Title: {n.get('title')}\nSource: {n.get('source')}"
                    for n in research["recent_news"][:3]
                ])
 
        cache_path  = os.path.join(
            email_cache_folder, f"{safe_filename}_{service_focus.lower()}.json"
        )
        full_prompt = _build_email_prompt(
            company_name, industry, financial_intel,
            market_news, pain_points_str, service_focus,
        )
 
        await queue.put({
            "company":     company_name,
            "index":       index,
            "prompt":      full_prompt,
            "cache_path":  cache_path,
            "retry_count": 0,
        })
 
    worker_coros = [
        _email_worker_loop(
            worker_id=i,
            key_worker=w,
            queue=queue,
            results=results,
            total_expected=len(failed_indices),
            email_cache_folder=email_cache_folder,
            service_focus=service_focus,
            worker_pool=retry_workers,
        )
        for i, w in enumerate(retry_workers)
    ]
    await asyncio.gather(*worker_coros, return_exceptions=True)
 
    fixed = 0
    still_failed = []   # companies that failed even Cerebras+Groq retry
 
    for index, res in results.items():
        raw_email  = res.get("raw_email", "ERROR")
        source     = res.get("source", "Failed")
        cache_path = res.get("cache_path", "")
 
        subject_line, email_body = _parse_email_output(raw_email)
 
        if subject_line and email_body and "ERROR" not in raw_email:
            df_output.at[index, "Generated_Email_Subject"] = subject_line
            df_output.at[index, "Generated_Email_Body"]    = email_body
            df_output.at[index, "AI_Source"]               = f"{source}(retry)"
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"subject": subject_line, "body": email_body, "source": source},
                        f, indent=4,
                    )
            fixed += 1
        else:
            # Still failed after Cerebras+Groq retry — collect for Azure fallback
            still_failed.append({
                "index":      index,
                "prompt":     res.get("prompt", ""),
                "cache_path": cache_path,
                "company":    res.get("company", ""),
            })
 
    # ── AZURE FALLBACK — for companies that failed ALL previous attempts ──
    # 4 parallel Azure workers — fast, no rate limit contention with main pipeline
    # Called ONLY here, so Azure is never touched during main pipeline run.
    azure_fixed = 0
    if still_failed:
        logging.warning(
            f"\n🔵 AZURE FALLBACK — {len(still_failed)} companies still failed after retry.\n"
            f"   Launching 4 parallel Azure workers now...\n"
        )
 
        azure_queue = asyncio.Queue()
        for item in still_failed:
            await azure_queue.put(item)
 
        async def _azure_worker(worker_num: int):
            nonlocal azure_fixed
            while True:
                try:
                    item = azure_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
 
                index      = item["index"]
                prompt     = item.get("prompt", "")
                cache_path = item["cache_path"]
                company    = item["company"]
 
                # Rebuild prompt if missing
                if not prompt:
                    try:
                        row          = original_df.loc[index]
                        company_name = str(row.get("Company Name", "")).strip()
                        industry     = str(row.get("Industry", "Technology"))
                        financial_intel = (
                            f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
                            f"Total Funding: {row.get('Total Funding', 'N/A')}"
                        )
                        safe_filename = (
                            "".join(c for c in company_name if c.isalnum() or c in "._- ")
                            .strip().replace(" ", "_").lower()
                        )
                        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
                        pain_points_str = "Not available."
                        market_news     = "No recent market updates available."
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                research = json.load(f)
                            if "pain_points" in research:
                                pain_points_str = "\n".join([f"- {p}" for p in research["pain_points"]])
                            if "recent_news" in research:
                                market_news = "\n---\n".join([
                                    f"Title: {n.get('title')}\nSource: {n.get('source')}"
                                    for n in research["recent_news"][:3]
                                ])
                        prompt = _build_email_prompt(
                            company_name, industry, financial_intel,
                            market_news, pain_points_str, service_focus,
                        )
                    except Exception as rebuild_err:
                        logging.error(f"❌ [AZURE W{worker_num}] Could not rebuild prompt for {company}: {rebuild_err}")
                        azure_queue.task_done()
                        continue
 
                try:
                    raw_email = await asyncio.wait_for(call_azure_async(prompt), timeout=45.0)
                    raw_email = raw_email or "ERROR: Azure empty response"
                    subject_line, email_body = _parse_email_output(raw_email)
 
                    # Loose parse fallback — raw output better than blank
                    if not subject_line or not email_body or "ERROR" in raw_email:
                        lines = [l.strip() for l in raw_email.strip().split('\n') if l.strip()]
                        subject_line = lines[0].replace("Subject:", "").strip() if lines else "Follow Up"
                        email_body   = '\n'.join(lines[1:]).strip() if len(lines) > 1 else raw_email
                        logging.warning(f"⚠️ [AZURE W{worker_num}] {company} — Strict parse failed, saving raw output.")
 
                    if subject_line and email_body:
                        df_output.at[index, "Generated_Email_Subject"] = subject_line
                        df_output.at[index, "Generated_Email_Body"]    = email_body
                        df_output.at[index, "AI_Source"]               = "Azure(fallback)"
                        if cache_path:
                            with open(cache_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {"subject": subject_line, "body": email_body, "source": "Azure"},
                                    f, indent=4,
                                )
                        logging.info(f"✅ [AZURE W{worker_num}] {company} — Done.")
                        azure_fixed += 1
                    else:
                        logging.error(f"❌ [AZURE W{worker_num}] {company} — Azure returned empty. Leaving blank.")
 
                except Exception as azure_err:
                    logging.error(f"❌ [AZURE W{worker_num}] {company} — Azure error: {azure_err}")
 
                azure_queue.task_done()
 
        # 4 parallel Azure workers
        await asyncio.gather(*[_azure_worker(i) for i in range(4)])
 
    logging.info(
        f"\n🔁 RETRY COMPLETE\n"
        f"   Fixed (Cerebras/Groq retry) : {fixed}\n"
        f"   Fixed (Azure fallback)       : {azure_fixed}\n"
        f"   Still blank                  : {len(still_failed) - azure_fixed}\n"
    )
    return df_output
 
 
# ==============================================================================
# ASYNC RUNNER
# ==============================================================================
 
async def _async_email_runner(
    df:                 pd.DataFrame,
    json_data_folder:   str,
    service_focus:      str,
    email_cache_folder: str,
) -> pd.DataFrame:
    """
    Queue-based async engine with 18 parallel workers.
 
    Architecture:
      ┌─────────────────────────────────────────────────────────┐
      │  asyncio.Queue  ←  all pending email tasks              │
      │                                                         │
      │  9 Gemini workers  ──┐                                  │
      │  3 Cerebras workers ─┼──► compete on the same queue     │
      │  6 Groq workers   ──┘                                   │
      │                                                         │
      │  Each worker owns one API key + enforces its own timing │
      └─────────────────────────────────────────────────────────┘
 
    Speed (500 emails, all 18 workers):
      Gemini   9 × 10/min =  90/min
      Cerebras 3 × 21/min =  63/min
      Groq     6 ×  9/min =  54/min
      ─────────────────────────────
      Total              = 207/min  →  ~2.5 min (realistic: 3–5 min)
    """
    # BUG FIX 2: Reset circuit breaker globals at the start of every call.
    # These are module-level globals. If a previous batch tripped the breaker,
    # every subsequent call to _async_email_runner would exit immediately
    # without processing any tasks — causing silent data loss across batches.
    global CONSECUTIVE_FAILURES, CIRCUIT_BREAKER_TRIPPED
    CONSECUTIVE_FAILURES    = 0
    CIRCUIT_BREAKER_TRIPPED = False
 
    os.makedirs(email_cache_folder, exist_ok=True)
 
    df_output = df.copy()
    df_output["Generated_Email_Subject"] = ""
    df_output["Generated_Email_Body"]    = ""
    df_output["AI_Source"]               = ""
 
    try:
        worker_pool = build_worker_pool()
    except RuntimeError as e:
        logging.critical(str(e))
        raise
 
    queue          = asyncio.Queue()
    tasks_to_run   = []
    processed_companies = {}
 
    for index, row in df_output.iterrows():
        # UPDATED: Only read required CSV columns
        company_name  = str(row.get("Company Name", "")).strip()
        industry      = str(row.get("Industry", "Technology")).strip()
        
        safe_filename = (
            "".join(c for c in company_name if c.isalnum() or c in "._- ")
            .strip()
            .replace(" ", "_")
            .lower()
        )
        cache_path = os.path.join(
            email_cache_folder, f"{safe_filename}_{service_focus.lower()}.json"
        )
 
        # Duplicate company check
        if company_name in processed_companies:
            prev_cache = processed_companies[company_name]
            if os.path.exists(prev_cache):
                with open(prev_cache, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                df_output.at[index, "Generated_Email_Subject"] = cached.get("subject", "")
                df_output.at[index, "Generated_Email_Body"]    = cached.get("body", "")
                df_output.at[index, "AI_Source"]               = "Cache(same-company)"
                logging.info(f"⏩ Duplicate company reuse: {company_name}")
            continue
 
        # Cache check
        if os.path.exists(cache_path):
            logging.info(f"⏩ Cache hit: {company_name}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            df_output.at[index, "Generated_Email_Subject"] = cached.get("subject", "")
            df_output.at[index, "Generated_Email_Body"]    = cached.get("body",    "")
            df_output.at[index, "AI_Source"]               = cached.get("source",  "Cache")
            continue
 
        # Load research data
        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        pain_points_str = "Not available."
        market_news     = "No recent market updates available."
 
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                research = json.load(f)
            if "pain_points" in research:
                pain_points_str = "\n".join(
                    [f"- {p}" for p in research["pain_points"]]
                )
            if "recent_news" in research:
                market_news = "\n---\n".join([
                    f"Title: {n.get('title')}\nSource: {n.get('source')}"
                    for n in research["recent_news"][:3]
                ])
                logging.info(f"📊 News loaded for {company_name}")
 
        # UPDATED: Only read Annual Revenue and Total Funding
        financial_intel = (
            f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
            f"Total Funding: {row.get('Total Funding', 'N/A')}"
        )
 
        logging.info(
            f"\n{'='*48}\n"
            f"PROMPT INPUT\n"
            f"  Company    : {company_name}\n"
            f"  Industry   : {industry}\n"
            f"  Financials : {financial_intel}\n"
            f"  News lines : :\n{market_news}\n"
            f"  Pains      : :\n{pain_points_str}\n"
            f"{'='*48}"
        )
 
        full_prompt = _build_email_prompt(
            company_name, industry, financial_intel,
            market_news, pain_points_str, service_focus,
        )
 
        task = {
            "company":     company_name,
            "index":       index,
            "prompt":      full_prompt,
            "cache_path":  cache_path,
            "retry_count": 0,
        }
        tasks_to_run.append(task)
        processed_companies[company_name] = cache_path
 
    if not tasks_to_run:
        logging.info("✅ All emails already cached — nothing to process.")
        return df_output
 
    total_expected = len(tasks_to_run)
    logging.info(
        f"\n🚀 PIPELINE START\n"
        f"   Emails to generate : {total_expected}\n"
        f"   Workers launched   : {len(worker_pool)}\n"
        f"   Estimated time     : ~{max(1, total_expected // 200)} – "
        f"{max(2, total_expected // 150)} minutes\n"
    )
 
    for task in tasks_to_run:
        await queue.put(task)
 
    results: dict = {}
 
    worker_coros = [
        _email_worker_loop(
            worker_id=i,
            key_worker=w,
            queue=queue,
            results=results,
            total_expected=total_expected,
            email_cache_folder=email_cache_folder,
            service_focus=service_focus,
            worker_pool=worker_pool,
        )
        for i, w in enumerate(worker_pool)
    ]
    await asyncio.gather(*worker_coros, return_exceptions=True)
 
    # BUG FIX 5: If workers all exited but tasks still remain unprocessed
    # (e.g. all keys exhausted mid-run), drain the queue and log missing companies.
    # This prevents silent data loss — user gets all successfully built emails,
    # and missing ones are clearly logged so they can be retried.
    remaining_in_queue = queue.qsize()
    if remaining_in_queue > 0:
        logging.warning(
            f"⚠️  PIPELINE INCOMPLETE: {remaining_in_queue} tasks still in queue after all workers exited."
            f" This usually means all API keys were exhausted. Returning {len(results)}/{total_expected} emails."
        )
        while not queue.empty():
            try:
                leftover = queue.get_nowait()
                logging.warning(f"   ↳ Not processed: {leftover.get('company', 'unknown')}")
                queue.task_done()
            except Exception:
                break
 
    success_count = 0
    fail_count    = 0
 
    for index, res in results.items():
        raw_email  = res.get("raw_email", "ERROR")
        source     = res.get("source",    "Failed")
        cache_path = res.get("cache_path","")
 
        subject_line, email_body = _parse_email_output(raw_email)
 
        df_output.at[index, "Generated_Email_Subject"] = subject_line
        df_output.at[index, "Generated_Email_Body"]    = email_body
        df_output.at[index, "AI_Source"]               = source
 
        if subject_line and email_body and "ERROR" not in raw_email and cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"subject": subject_line, "body": email_body, "source": source},
                    f, indent=4,
                )
            success_count += 1
        else:
            fail_count += 1
 
    df_output = await _retry_failed_emails(
        df_output=df_output,
        original_df=df.copy(),
        json_data_folder=json_data_folder,
        service_focus=service_focus,
        email_cache_folder=email_cache_folder,
        worker_pool=worker_pool,
    )
 
    source_counts: dict = {}
    for res in results.values():
        s = res.get("source", "Unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
 
    final_success = df_output.shape[0]
    final_failed  = total_expected - final_success
 
    logging.info(
        f"\n{'='*48}\n"
        f"PIPELINE COMPLETE\n"
        f"  Total processed : {total_expected}\n"
        f"  Main pipeline   : {success_count} success, {fail_count} failed\n"
        f"  After retry     : {final_success} success, {final_failed} failed\n"
        f"  By source       : {source_counts}\n"
        f"{'='*48}\n"
    )
 
    return df_output
 
 
# ==============================================================================
# SYNCHRONOUS WRAPPER
# ==============================================================================
 
def run_serpapi_email_generation(
    df:                 pd.DataFrame,
    json_data_folder:   str  = "research_cache",
    service_focus:      str  = "salesforce",
    email_cache_folder: str  = "email_cache",
) -> pd.DataFrame:
    """
    Synchronous entry point.
    Safe to call from Streamlit, Jupyter, or any non-async context.
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
 
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
 
    return loop.run_until_complete(
        _async_email_runner(df, json_data_folder, service_focus, email_cache_folder)
    )
 
 
# ==============================================================================
# STANDALONE ENTRY POINT
# ==============================================================================
 
if __name__ == "__main__":
    logging.info("🚀 Running 3-API async pipeline (Gemini + Cerebras + Groq)…")
 
    CSV_FILE_PATH   = r"C:\Users\user\Desktop\Solution_Reverse_Enginnring\500_deployement - Copy\IT_Services_Filtered - Sheet9 (5).csv"
    TXT_OUTPUT_FILE = "standalone_generated_email_IT_gemini_serp777.txt"
    LOCAL_SERVICE_MODE = "AI"
 
    try:
        if os.path.exists(CSV_FILE_PATH):
 
            df = pd.read_csv(CSV_FILE_PATH)
            logging.info(f"Total rows in dataset: {len(df)}")
 
            df["Industry"] = (
                df["Industry"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
 
            filtered_df = df[df["Industry"] == "information technology & services"]
            logging.info(f"IT Services companies found: {len(filtered_df)}")
 
            if len(filtered_df) == 0:
                logging.error("❌ No IT Services companies found.")
                sys.exit(1)
 
            test_df = filtered_df.sample(
                n=min(10, len(filtered_df)),
                random_state=None,
            ).reset_index(drop=True)
            logging.info(f"Selected {len(test_df)} companies for test run.")
 
            result_df = run_serpapi_email_generation(
                test_df, service_focus=LOCAL_SERVICE_MODE
            )
 
            with open(TXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
                for _, row in result_df.iterrows():
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write(
                        f"COMPANY: {row.get('Company Name', 'Unknown')} | "
                        f"INDUSTRY: {row.get('Industry', 'Unknown')} | "
                        f"SOURCE: {row.get('AI_Source', 'Unknown')}\n"
                    )
                    f.write("=" * 60 + "\n\n")
                    f.write(f"SUBJECT: {row.get('Generated_Email_Subject', '')}\n\n")
                    f.write(str(row.get("Generated_Email_Body", "")))
                    f.write("\n\n")
 
            logging.info(f"✅ Done. Emails saved to: {TXT_OUTPUT_FILE}")
 
        else:
            logging.error("❌ CSV file not found.")
 
    except Exception as e:
        logging.critical(f"❌ Standalone execution error: {e}")
