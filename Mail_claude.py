






import os
import json
import pandas as pd
import asyncio
import re
import sys
import logging
import time
import threading
import unicodedata 
from google import genai
from google.genai import types
from groq import AsyncGroq
from cerebras.cloud.sdk import AsyncCerebras
import tiktoken
from openai import AsyncAzureOpenAI
from api_rotating_claude import (
    KeyWorker,      build_worker_pool,
    get_azure_config,
)
 
####
#tiktoken setup
_ENC = tiktoken.get_encoding("cl100k_base")

def _tok(text: str) -> int:
    """Count tokens in any text string."""
    try:
        return len(_ENC.encode(str(text)))
    except Exception:
        return len(str(text)) // 4   # fallback estimate
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

def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = "".join(c for c in name if c.isalnum() or c in "._- ")
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r"_+", "_", name)
    return name 
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
 
# threading.Lock — safe across all threads and event loops
_cb_lock = threading.Lock()
 
 
# ==============================================================================
# API CALL FUNCTIONS
# ==============================================================================
 
# ==============================================================================
# SYSTEM PROMPT — fixed persona, injected via system role on every call
# Keeps user prompt focused on company data only → better quality per token
# ==============================================================================
 
SYSTEM_PROMPT = """You are a senior B2B sales copywriter at AnavClouds with 12 years writing cold outbound for enterprise tech companies. You've written thousands of emails. You know what gets replies and what goes to spam.
 
WRITING STYLE:
- Write like a busy, sharp professional — short sentences, real observations, zero fluff
- Never write marketing copy. Write peer-to-peer business notes.
- Use contractions naturally (don't, we're, it's, they've)
- Sentences are uneven in length — that's intentional
- Never start with "I wanted to" and never end with a question or CTA
- Notice one specific thing about the company and react to it — not summarize it
 
OUTPUT DISCIPLINE:
- Follow the exact format given — no extra sections, no sign-offs
- Stop writing immediately after the 4th bullet
- Never use banned words even once — if you catch yourself, rewrite
- Never produce symmetric bullets — each one feels different in length and style
 
FORBIDDEN PHRASES (rewrite any sentence containing these):
reach out, touch base, circle back, game-changer, cutting-edge, best-in-class, world-class,
I wanted to connect, Hope this finds you well, Let me know if you're interested, Would love to,
Excited to share, Scale your business, Drive results, Unlock potential, Quick call, Hop on a call,
Free consultation, Revolutionize, Transform, Disrupt, Just checking in
 
BANNED WORDS (not even once):
accelerate, certified, optimize, enhance, leverage, synergy, streamline, empower, solutions,
deliverables, bandwidth, mission-critical, investment, fast, new, Here
 
HARD RULES:
- NO exclamation marks
- NO all-caps
- NO CTA
- NO sign-off
- NO ending question
- Email stops immediately after bullet 4. Nothing after it.
- Subject format: [Desired Outcome] without [Core Friction] — no tools/services/buzzwords"""
 
# async def call_gemini_async(prompt: str, api_key: str) -> str:
#     """
#     Google Gemini 2.5 Flash — PRIMARY
#     System instruction used for persona, user prompt for company data.
#     """
#     client   = genai.Client(api_key=api_key)
#     response = await client.aio.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.25,
#             max_output_tokens=2000,
#             system_instruction=SYSTEM_PROMPT,
#         ),
#     )
#     return response.text

async def call_gemini_async(prompt: str, api_key: str) -> str:
    # COUNT INPUT TOKENS before sending
    sys_tok    = _tok(SYSTEM_PROMPT)
    prompt_tok = _tok(prompt)
    input_tok  = sys_tok + prompt_tok

    client   = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            max_output_tokens=2500,
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    # COUNT OUTPUT TOKENS after receiving
    output_tok = _tok(response.text or "")
    total_tok  = input_tok + output_tok

    logging.info(
        f"[Gemini] system={sys_tok} + prompt={prompt_tok} + output={output_tok} "
        f"= TOTAL {total_tok} tokens"
    )
    return response.text
 
 
# async def call_cerebras_async(prompt: str, api_key: str) -> str:
#     """
#     Cerebras llama3.1-8b — SECONDARY
#     System role used for persona, user role for company task.
#     """
#     client   = AsyncCerebras(api_key=api_key)
#     response = await client.chat.completions.create(
#         model="llama3.1-8b",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": prompt},
#         ],
#         temperature=0.25,
#         max_completion_tokens=2000,
#     )
#     return response.choices[0].message.content

async def call_cerebras_async(prompt: str, api_key: str) -> str:
    # COUNT INPUT TOKENS before sending
    sys_tok    = _tok(SYSTEM_PROMPT)
    prompt_tok = _tok(prompt)
    input_tok  = sys_tok + prompt_tok

    client   = AsyncCerebras(api_key=api_key)
    response = await client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.25,
        max_completion_tokens=2100,
    )

    choice     = response.choices[0]
    content    = choice.message.content or ""

    # COUNT OUTPUT TOKENS after receiving
    output_tok = _tok(content)
    total_tok  = input_tok + output_tok

    logging.info(
        f"[Cerebras] system={sys_tok} + prompt={prompt_tok} + output={output_tok} "
        f"= TOTAL {total_tok} tokens"
    )

    if getattr(choice, "finish_reason", None) == "length":
        return "ERROR: Cerebras cut output mid-sentence (finish_reason=length). Needs retry."
    return content
 
 
# async def call_groq_async(prompt: str, api_key: str) -> str:
#     """
#     Groq llama-4-scout-17b — OVERFLOW
#     System role used for persona, user role for company task.
#     """
#     client   = AsyncGroq(api_key=api_key)
#     response = await client.chat.completions.create(
#         model="meta-llama/llama-4-scout-17b-16e-instruct",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": prompt},
#         ],
#         temperature=0.25,
#         max_tokens=2000,
#     )
#     return response.choices[0].message.content

async def call_groq_async(prompt: str, api_key: str) -> str:
    sys_tok    = _tok(SYSTEM_PROMPT)
    prompt_tok = _tok(prompt)
    input_tok  = sys_tok + prompt_tok

    client   = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.25,
        max_tokens=2100,
    )

    choice     = response.choices[0]
    content    = choice.message.content or ""
    output_tok = _tok(content)
    total_tok  = input_tok + output_tok

    logging.info(
        f"[Groq] system={sys_tok} + prompt={prompt_tok} + output={output_tok} "
        f"= TOTAL {total_tok} tokens"
    )

    if getattr(choice, "finish_reason", None) == "length":
        return "ERROR: Groq cut output mid-sentence (finish_reason=length). Needs retry."
    return content
 
# async def call_azure_async(prompt: str) -> str:
#     """
#     Azure OpenAI GPT-4o Mini — EMERGENCY FALLBACK
#     """
#     config = get_azure_config()
 
#     client = AsyncAzureOpenAI(
#         api_key        = config["api_key"],
#         azure_endpoint = config["endpoint"],
#         api_version    = config["api_version"],
#     )
 
#     response = await client.chat.completions.create(
#         model       = config["deployment"],
#         messages    = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": prompt},
#         ],
#         temperature = 0.25,
#         max_tokens  = 2000,
#     )
#     return response.choices[0].message.content


async def call_azure_async(prompt: str) -> str:
    sys_tok    = _tok(SYSTEM_PROMPT)
    prompt_tok = _tok(prompt)
    input_tok  = sys_tok + prompt_tok

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
        max_tokens  = 2100,
    )

    content    = response.choices[0].message.content or ""
    output_tok = _tok(content)
    total_tok  = input_tok + output_tok

    logging.info(
        f"[Azure] system={sys_tok} + prompt={prompt_tok} + output={output_tok} "
        f"= TOTAL {total_tok} tokens"
    )
    return content
 
# ==============================================================================
# SERVICE CAPABILITY BLOCKS — only relevant one injected per prompt
# ==============================================================================
 
_SERVICE_BLOCK_AI = """
* Build enterprise AI agents and AI copilots using Generative AI, custom LLMs, RAG pipelines, and vector databases to unlock insights from enterprise data.
* Develop predictive machine learning models for lead scoring, demand forecasting, churn prediction, and intelligent decision-making.
* Design modern data platforms with scalable ETL/ELT pipelines, data lakes, and cloud data warehouses for real-time analytics and reporting.
* Implement advanced AI solutions including Agentic AI systems, conversational AI assistants, and AI-powered automation to improve operational efficiency.
* Enable AI-driven business intelligence with predictive analytics dashboards and data-driven insights for leadership teams.
* Automate complex workflows using Python, AI frameworks, and orchestration tools to reduce manual effort and increase productivity.
* Implement MLOps and LLMOps frameworks to ensure reliable, scalable, and secure AI model deployment."""
 
_SERVICE_BLOCK_SALESFORCE = """
* Implement and customize Salesforce platforms including Sales Cloud, Service Cloud, Marketing Cloud, Experience Cloud, and Industry Clouds.
* Deploy AI-powered CRM capabilities using Salesforce Data Cloud, Einstein AI, and Agentforce for intelligent automation and insights.
* Develop scalable Salesforce solutions using Apex, Lightning Web Components (LWC), and Flow automation to improve operational efficiency.
* Integrate Salesforce with ERP systems, marketing platforms, and enterprise applications using MuleSoft and modern APIs.
* Implement Revenue Cloud and CPQ solutions to streamline quoting, pricing, and revenue management processes.
* Automate marketing and customer journeys using Salesforce Marketing Cloud and Account Engagement (Pardot).
* Integrate Salesforce with Slack, Tableau, and analytics platforms to improve collaboration and real-time reporting.
* Provide 24/7 Salesforce managed services including admin support, system monitoring, optimization, and proactive health checks."""

# _SERVICE_BLOCK_COMBINED is not used as a single block anymore.
# For combined mode, _build_combined_email_prompt() uses both AI and Salesforce blocks separately.
_SERVICE_BLOCK_COMBINED = _SERVICE_BLOCK_SALESFORCE  # fallback placeholder, not used directly
 
_SERVICE_BLOCKS = {
    "ai":         _SERVICE_BLOCK_AI,
    "salesforce": _SERVICE_BLOCK_SALESFORCE,
    "combined":   _SERVICE_BLOCK_COMBINED,
}
 
# ==============================================================================
# PROMPT BUILDER
# ==============================================================================
 
def _build_combined_email_prompt(
    company:     str,
    industry:    str,
    financials:  str,
    market_news: str,
    pain_points: str,
) -> str:
    """
    Prompt for combined (Salesforce + AI) service focus.
    Email structure after the transition line lists both service sections separately:
      Salesforce services—
      1. ...  2. ...  3. ...  4. ...
      AI services—
      1. ...  2. ...  3. ...  4. ...
    """
    sf_caps = _SERVICE_BLOCK_SALESFORCE
    ai_caps = _SERVICE_BLOCK_AI

    return f"""
SELL: Both Salesforce and AI services together. Mention "AnavClouds" once, in Block 2 only.

COMPANY DATA:
- Company: {company}
- Industry: {industry}
- Financials: {financials}
- Market News: {market_news}
- Pain Points: {pain_points}

---
THINK BEFORE WRITING (internal only — do not output):
1. Extract ONE strong signal from market_news or financials (Growth / Operational / Tech / GTM).
2. Pick the 2 strongest pains — one that maps to Salesforce work, one that maps to AI/data work.
3. Frame each pain as an outcome phrase (what good looks like, not what's broken).
4. Draft Block 1 opener. Ask: does it sound like you read about them this morning? Rewrite until yes.

IMPORTANT: You MUST write the complete email including all 8 bullets. 
Do NOT stop before completing AI services section.
---
OUTPUT FORMAT (follow exactly — no deviations):

SUBJECT:
[One line. Outcome without Friction. No tools, no buzzwords, no company name.]

Hi ,

[Block 1 — exactly 2 lines. Write both lines together as one paragraph — NO line break between them. Both lines together must be 180 to 200 characters total.
line 1: Start with "I noticed" or "I saw". Reference ONE specific news item or financial signal. React like a peer — don't summarize, don't explain. One sharp observation only.
line 2: Connect to a natural business direction. No pain mention. No industry name. No generic sector statements.]

[Block 2 — 2 lines only.
Line 1: ALWAYS start with "At AnavClouds," — describe what we do as the logical next layer for where this company is heading. Mention both Salesforce and AI/data work naturally in prose. Never bullet here.
Line 2: "We've helped teams [outcome of Salesforce pain] and [outcome of AI pain]." — mapped directly to THIS company's pain points, not generic.]

[Pick ONE transition randomly, end with colon:
"Here are some ways we can help:"
"Here's what usually helps in situations like this :"
"A few practical ways teams simplify this :"
"What tends to work well in cases like this :"
"Here's what teams often find useful :"]

Salesforce services—
• [How we help fix their biggest CRM, sales, or customer management problem — written as a plain outcome anyone can understand.]
• [A specific improvement to their sales process, customer workflows, or team operations — different angle from bullet 1.]
• [How we solve a second Salesforce-related pain — framed as a result they get, not a service we offer.]
• [One concrete Salesforce capability from the list below that fits this company's situation — keep it simple and     outcome-focused.]

SALESFORCE CAPABILITIES (pick from these, rewrite in plain English):
{sf_caps}

AI services—
• [How we help fix their biggest data, automation, or decision-making problem — written as a plain outcome anyone can understand.]
• [A specific improvement to their reporting, predictions, or workflow automation — different angle from bullet 1.]
• [How we solve a second AI-related pain — framed as a result they get, not a service we offer.]
• [One concrete AI capability from the list below that fits this company's situation — keep it simple and outcome-focused.]

AI CAPABILITIES (pick from these, rewrite in plain English):
{ai_caps}

BULLET LANGUAGE RULE: Every bullet must be written in plain English. The reader is a CEO with zero technical background. No tool names, no acronyms — focus only on the business outcome.

BULLET END RULE: Every bullet MUST end with a period (.). No exceptions.

SPACING RULES:
- After "Hi ," → exactly ONE blank line before Block 1
- After transition line → exactly ONE blank line before "Salesforce services—"
- After "Salesforce services—" → exactly ONE blank line before first bullet
- After last Salesforce bullet → exactly ONE blank line before "AI services—"
- After "AI services—" → exactly ONE blank line before first bullet
- NO blank lines between bullets within a section

Strictly Follow: You MUST write the complete email including all 8 bullets. 
Do NOT stop before completing AI services section.
FINAL CHECK before outputting:
- Subject line is ONE line only?
- Block 2 starts with "At AnavClouds,"?
- Both sections present: "Salesforce services—" and "AI services—"?
- Exactly 4 • bullets under each section?
- No numbered list anywhere (no 1. 2. 3. 4.)?
- No CTA anywhere?
- Ends after last AI bullet with no sign-off?
→ If all yes, output.
"""


def _build_email_prompt(
    company:    str,
    industry:   str,
    financials: str,
    market_news:str,
    pain_points:str,
    service_focus: str,
) -> str:
    """
    Optimized prompt — system prompt handles persona, user prompt handles task.
    For combined service_focus, routes to _build_combined_email_prompt() which
    generates a two-section email (Salesforce services + AI services).
    """

    # Route combined to its own dedicated prompt structure
    if service_focus.lower() == "combined":
        return _build_combined_email_prompt(
            company, industry, financials, market_news, pain_points
        )

    capabilities = _SERVICE_BLOCKS.get(service_focus.lower(), _SERVICE_BLOCK_AI)
    

    return f"""
SELL: {service_focus} only. Mention "AnavClouds" once, in Block 2 only.
 
COMPANY DATA:
- Company: {company}
- Industry: {industry}
- Financials: {financials}
- Market News: {market_news}
- Pain Points: {pain_points}
 
CAPABILITIES TO USE:
{capabilities}
 
---

IMPORTANT: Write the COMPLETE email. Do NOT stop before the 4th bullet.

THINK BEFORE WRITING (internal only — do not output):
1. Extract ONE strong signal from market_news or financials (Growth / Operational / Tech / GTM).
2. Pick the 2 strongest pains. Convert each to an outcome phrase (what good looks like, not what's broken).
3. Map those pains to the capabilities above. Frame as outcomes, not features. Tone: curious peer, not vendor.
4. Draft Block 1 opener. Ask yourself: does it sound like you read about them this morning? Rewrite until yes.



----
OUTPUT FORMAT (follow exactly — no deviations):
 
SUBJECT:[One line. Outcome without Friction. No tools, no buzzwords, no company name.]
 
Hi ,
[BLANK LINE HERE — mandatory empty line after greeting before Block 1]

[[Block 1 — exactly 2 lines. Write both lines together as one paragraph — NO line break between them. Both lines together must be 180 to 200 characters total.
line 1: Start with "I noticed" or "I saw". Reference ONE specific news item or financial signal. React like a peer — don't summarize, don't explain. One sharp observation only.
line 2: Connect to a natural business direction. No pain mention. No industry name. No generic sector statements.]
 
[Block 2 — 2 lines only.
Line 1: ALWAYS start with "At AnavClouds," — describe what we do as the logical next layer for where this company is heading. Mention 2-3 work areas naturally in prose. Never bullet here.
Line 2: "We've helped teams [outcome of pain 1] and [outcome of pain 2]." — mapped directly to THIS company's pain points, not generic.]
 
[Pick ONE transition randomly, end with colon:
"Here are some ways we can help:"
"Here's what usually helps in situations like this :"
"A few practical ways teams simplify this :"
"What tends to work well in cases like this :"
"Here's what teams often find useful :"]
 
• [Bullet 1 — direct fix for strongest pain. Outcome-framed. Conversational, not polished.]
 
• [Bullet 2 — broader {industry} workflow, data setup, or tech debt improvement. Different length from bullet 1.]
 
• [Bullet 3 — fix for second pain. Framed as result, not as a service being offered.]
 
• [Bullet 4 — one specific {service_focus} technical method or architecture tied directly to {industry}. Must feel specialist-level. Never generic. Never staffing. Never RAG as default.]
 
BULLET RULES: blank line after transition colon, blank line between each bullet, use only •, no symmetry, no marketing copy.

MUST COMPLETE: All 4 bullets must be written before stopping.

FINAL CHECK before outputting:
- No banned word used?
- Block 2 starts with "At AnavClouds,"?
- Bullet 4 is technical, not staffing?
- No CTA anywhere?
- Ends after last bullet with no sign-off?
→ If all yes, output.
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
            # Python 3.14 FIX: asyncio.wait_for() requires a running Task context.
            # asyncio.wait() on a Future works outside of Tasks too.
            _get_fut = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({_get_fut}, timeout=5.0)
            if not done:
                _get_fut.cancel()
                if len(results) >= total_expected:
                    break
                continue
            task = _get_fut.result()
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
            # Python 3.14 FIX: use ensure_future + asyncio.wait instead of wait_for
            if key_worker.provider == "gemini":
                _api_fut = asyncio.ensure_future(call_gemini_async(full_prompt, key_worker.api_key))
            elif key_worker.provider == "cerebras":
                _api_fut = asyncio.ensure_future(call_cerebras_async(full_prompt, key_worker.api_key))
            else:
                _api_fut = asyncio.ensure_future(call_groq_async(full_prompt, key_worker.api_key))

            done, _ = await asyncio.wait({_api_fut}, timeout=35.0)
            if not done:
                _api_fut.cancel()
                raise asyncio.TimeoutError()
            raw_email = _api_fut.result()
            
            raw_email = raw_email or "ERROR: API returned empty response"
 
            with _cb_lock:
                CONSECUTIVE_FAILURES = 0
 
            key_worker.reset_retry_count()  
 
            # subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
            # if subject_line and email_body and "ERROR" not in raw_email:
                # if cache_path:
                #     with open(cache_path, "w", encoding="utf-8") as f:
                #         json.dump({"subject": subject_line, "body": email_body, "source": provider_label}, f, indent=4)
            #     logging.info(f"✅ [W{worker_id:02d}|{provider_label}] {company} — Done & Cached.")
            # else:
            #     logging.warning(f"⚠️  [W{worker_id:02d}|{provider_label}] {company} — Parsing Issue.")
            # subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
            # if subject_line and email_body and "ERROR" not in raw_email:
            subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
            subject_line = _clean_email_text(subject_line)   # ← ADD
            email_body   = _clean_email_text(email_body)     # ← ADD
            if subject_line and email_body and "ERROR" not in raw_email:
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"subject": subject_line, "body": email_body, "source": provider_label}, f, indent=4)
                logging.info(f"✅ [W{worker_id:02d}|{provider_label}] {company} — Done & Cached.")
            else:
                parse_error = email_body if email_body.startswith("ERROR") else "Unknown parse failure"
                logging.warning(
                    f"⚠️  [W{worker_id:02d}|{provider_label}] {company} — Parsing Issue.\n"
                    f"    REASON  : {parse_error}\n"
                    f"    RAW DUMP: {repr(raw_email[:1000])}"
                )
 
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
 
# def _parse_email_output(raw_email: str) -> tuple[str, str]:
#     if not raw_email:
#         return "", "ERROR: API returned empty response"
 
#     clean_text = raw_email.strip()
#     clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
#     clean_text = re.sub(r'\n```$', '', clean_text)
#     clean_text = clean_text.strip()
 
#     if clean_text.startswith("ERROR"):
#         return "", clean_text
 
#     subject_line = ""
#     email_body = clean_text
#     pre_body = ""
 
#     body_match = re.search(r'(?m)^((?:Hi|Hello|Hey|Dear)[\s,].*)', clean_text, re.DOTALL | re.IGNORECASE)
 
#     if body_match:
#         email_body = body_match.group(1).strip()
#         pre_body = clean_text[:body_match.start()].strip()
#     else:
#         parts = re.split(r'\n{2,}', clean_text, maxsplit=1)
#         if len(parts) == 2:
#             pre_body, email_body = parts[0].strip(), parts[1].strip()
#         else:
#             pre_body = ""
#             email_body = clean_text
 
#     if pre_body:
#         sub_clean = re.sub(r'(?i)\*?\*?SUBJECT:\*?\*?\s*', '', pre_body).strip()
#         sub_clean = re.sub(r'-\s*\n\s*', '', sub_clean)
#         sub_clean = re.sub(r'\s*\n\s*', ' ', sub_clean)
#         subject_line = re.sub(r'^"|"$', '', sub_clean).strip()
 
#     if subject_line and email_body.startswith(subject_line):
#         email_body = email_body[len(subject_line):].strip()
 
#     email_body = email_body.strip()
#     if not email_body:
#         return "", "ERROR: Email body is completely empty after parsing."
 
#     word_count = len(email_body.split())
#     if word_count < 40:
#         return "", f"ERROR: Truncated email (Only {word_count} words). Fails word count check."
 
#     bullet_matches = re.findall(r'(?m)^[\s]*[\*•\-\–\—]|â€¢', email_body)
#     # if len(bullet_matches) < 4:
#     #     return "", f"ERROR: Incomplete generation. Found only {len(bullet_matches)} bullets, expected 4."
#     if len(bullet_matches) < 2:
#         return "", f"ERROR: Too few bullets ({len(bullet_matches)})."
#     elif len(bullet_matches) < 4:
#         logging.warning(f"⚠️ Only {len(bullet_matches)} bullets — saving anyway.")
 
#     if email_body[-1] not in ['.', '!', '?', '"', '\'']:
#         return "", "ERROR: Email body cut off mid-sentence (No ending punctuation)."
 
#     last_line = email_body.split('\n')[-1].strip()
#     if len(last_line.split()) < 4 and last_line[-1] not in ['.', '!', '?']:
#         return "", f"ERROR: Last bullet point seems cut off ('{last_line}')."
 
#     return subject_line, email_body
def _clean_email_text(text: str) -> str:
    """
    Replace fancy Unicode characters with plain ASCII equivalents.
    Ensures bullet points and dashes render correctly in all email clients.
    """
    if not text:
        return text
    replacements = {
        "\u2022": "*",   # •  bullet point   → *
        "\u2014": "-",   # —  em dash        → -
        "\u2013": "-",   # –  en dash        → -
        "\u2018": "'",   # '  left quote     → '
        "\u2019": "'",   # '  right quote    → '
        "\u201c": '"',   # "  left dbl quote → "
        "\u201d": '"',   # "  right dbl quot → "
        "\u2026": "...", # …  ellipsis       → ...
        "\u00a0": " ",   # non-breaking space→ space
        "\x95":   "*",   # Windows bullet   → *
        "\x97":   "-",   # Windows em dash  → -
        "\x96":   "-",   # Windows en dash  → -
        "\x91":   "'",   # Windows left quote
        "\x92":   "'",   # Windows right quote
        "\x93":   '"',   # Windows left dbl quote
        "\x94":   '"',   # Windows right dbl quote
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

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
        sub_clean = re.sub(r'(?i)(\*?\*?SUBJECT:\*?\*?\s*)+', '', pre_body).strip()
        sub_clean = re.sub(r'-\s*\n\s*', '', sub_clean)
        sub_clean = re.sub(r'\s*\n\s*', ' ', sub_clean)
        subject_line = re.sub(r'^"|"$', '', sub_clean).strip()

    if subject_line and email_body.startswith(subject_line):
        email_body = email_body[len(subject_line):].strip()

    email_body = email_body.strip()
    if not email_body:
        return "", "ERROR: Email body is completely empty after parsing."

    # CHECK 1: Word count — truly empty or garbage emails only
    word_count = len(email_body.split())
    if word_count < 30:
        return "", f"ERROR: Too short ({word_count} words) — genuinely incomplete."

    # CHECK 2: Bullets — minimum 3 required, 4 is ideal
    # Less than 3 = AI clearly did not finish → retry worthy
    # Exactly 3 = acceptable, save it with warning
    bullet_matches = re.findall(r'(?m)^[\s]*[\*•\-\–\—]|â€¢', email_body)
    if len(bullet_matches) < 3:
        return "", f"ERROR: Only {len(bullet_matches)} bullets — genuinely incomplete, needs retry."
    if len(bullet_matches) == 3:
        logging.warning(f"⚠️ 3 bullets instead of 4 — saving anyway.")

    # CHECK 3: Ending punctuation — only warn, do NOT error
    # LLMs sometimes skip period on last bullet even when content is valid
    # if email_body[-1] not in ['.', '!', '?', '"', '\'']:
    #     logging.warning("⚠️ No ending punctuation — saving anyway.")

    SENTENCE_END = ('.', '!', '?', '"', "'", ')')
    all_lines    = [l.strip() for l in email_body.split('\n') if l.strip()]
    bullet_lines = [l for l in all_lines if re.match(r'^[•*\-\–\—]', l)]

    for i, line in enumerate(bullet_lines, start=1):
        content = re.sub(r'^[•*\-\–\—]\s*', '', line).strip()
        words   = content.split()
        if len(words) >= 4 and not content.endswith(SENTENCE_END):
            return "", (
                f"ERROR: Bullet {i} cut mid-sentence — no dot at end. "
                f"Snippet: '...{content[-50:]}' — needs retry."
            )

    # CHECK 4: Last line of entire email must also end properly
    last_line    = all_lines[-1] if all_lines else ""
    last_content = re.sub(r'^[•*\-\–\—]\s*', '', last_line).strip()
    if len(last_content.split()) >= 4 and not last_content.endswith(SENTENCE_END):
        return "", (
            f"ERROR: Email ends mid-sentence. "
            f"Last line: '...{last_content[-60:]}' — needs retry."
        )

    # CHECK 4: Last line cut off — REMOVED
    # This caused too many false positives on valid short bullets
    # Word count check above already catches truly truncated emails

    return subject_line, email_body


# def _parse_email_output_combined(raw_email: str) -> tuple[str, str]:
#     """
#     Parser for combined (Salesforce + AI) emails.
#     Validates subject, both section headers, 4 bullets each, no numbered lists.
#     """
#     if not raw_email:
#         return "", "ERROR: API returned empty response"

#     clean_text = raw_email.strip()
#     clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
#     clean_text = re.sub(r'\n```$', '', clean_text)
#     clean_text = clean_text.strip()

#     if clean_text.startswith("ERROR"):
#         return "", clean_text

#     subject_line = ""
#     email_body   = clean_text
#     pre_body     = ""

#     body_match = re.search(r'(?m)^((?:Hi|Hello|Hey|Dear)[\s,].*)', clean_text, re.DOTALL | re.IGNORECASE)
#     if body_match:
#         email_body = body_match.group(1).strip()
#         pre_body   = clean_text[:body_match.start()].strip()
#     else:
#         parts = re.split(r'\n{2,}', clean_text, maxsplit=1)
#         if len(parts) == 2:
#             pre_body, email_body = parts[0].strip(), parts[1].strip()
#         else:
#             pre_body   = ""
#             email_body = clean_text

#     if pre_body:
#         sub_clean    = re.sub(r'(?i)(\*?\*?SUBJECT:\*?\*?\s*)+', '', pre_body).strip()
#         sub_clean    = re.sub(r'-\s*\n\s*', '', sub_clean)
#         sub_clean    = re.sub(r'\s*\n\s*', ' ', sub_clean)
#         subject_line = re.sub(r'^"|"$', '', sub_clean).strip()

#     if not email_body:
#         return "", "ERROR: Email body is completely empty after parsing."

#     if len(email_body.split()) < 50:
#         return "", f"ERROR: Too short ({len(email_body.split())} words) — genuinely incomplete."

#     has_sf = bool(re.search(r'Salesforce services', email_body, re.IGNORECASE))
#     has_ai = bool(re.search(r'AI services',         email_body, re.IGNORECASE))
#     if not has_sf or not has_ai:
#         missing = []
#         if not has_sf: missing.append("Salesforce services—")
#         if not has_ai: missing.append("AI services—")
#         return "", f"ERROR: Missing section(s): {', '.join(missing)} — needs retry."

#     ai_split   = re.split(r'AI services[\u2014\-\u2013]?', email_body, flags=re.IGNORECASE)
#     sf_section = ai_split[0]
#     ai_section = ai_split[1] if len(ai_split) > 1 else ""

#     sf_bullets = re.findall(r'(?m)^[\s]*[\u2022*]', sf_section)
#     ai_bullets = re.findall(r'(?m)^[\s]*[\u2022*]', ai_section)

#     if len(sf_bullets) < 3:
#         return "", f"ERROR: Salesforce section has only {len(sf_bullets)} bullets (need 4) — needs retry."
#     if len(ai_bullets) < 3:
#         return "", f"ERROR: AI section has only {len(ai_bullets)} bullets (need 4) — needs retry."

#     numbered = re.findall(r'(?m)^\s*[1-4]\.\s', email_body)
#     if numbered:
#         return "", f"ERROR: Numbered list detected ({len(numbered)} items) — use • bullets only, needs retry."

#     SENTENCE_END = ('.', '!', '?', '"', "'", ')')
#     all_lines    = [l.strip() for l in email_body.split('\n') if l.strip()]
#     bullet_lines = [l for l in all_lines if re.match(r'^[\u2022*]', l)]

#     for i, line in enumerate(bullet_lines, start=1):
#         content = re.sub(r'^[\u2022*]\s*', '', line).strip()
#         if len(content.split()) >= 4 and not content.endswith(SENTENCE_END):
#             return "", f"ERROR: Bullet {i} cut mid-sentence — '...{content[-50:]}' — needs retry."

#     last_line    = all_lines[-1] if all_lines else ""
#     last_content = re.sub(r'^[\u2022*]\s*', '', last_line).strip()
#     if len(last_content.split()) >= 4 and not last_content.endswith(SENTENCE_END):
#         return "", f"ERROR: Email ends mid-sentence. Last line: '...{last_content[-60:]}' — needs retry."

#     return subject_line, email_body

def _parse_email_output_combined(raw_email: str) -> tuple[str, str]:
    """
    Parser for combined (Salesforce + AI) emails.
    Same logic as _parse_email_output() but expects 8 bullets total (4 SF + 4 AI)
    and validates both section headers are present.
    """
    if not raw_email:
        return "", "ERROR: API returned empty response"

    clean_text = raw_email.strip()
    clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
    clean_text = re.sub(r'\n```$', '', clean_text)
    clean_text = clean_text.strip()

    if clean_text.startswith("ERROR"):
        return "", clean_text

    subject_line = ""
    email_body   = clean_text
    pre_body     = ""

    body_match = re.search(r'(?m)^((?:Hi|Hello|Hey|Dear)[\s,].*)', clean_text, re.DOTALL | re.IGNORECASE)
    if body_match:
        email_body = body_match.group(1).strip()
        pre_body   = clean_text[:body_match.start()].strip()
    else:
        parts = re.split(r'\n{2,}', clean_text, maxsplit=1)
        if len(parts) == 2:
            pre_body, email_body = parts[0].strip(), parts[1].strip()
        else:
            pre_body   = ""
            email_body = clean_text

    if pre_body:
        sub_clean    = re.sub(r'(?i)(\*?\*?SUBJECT:\*?\*?\s*)+', '', pre_body).strip()
        sub_clean    = re.sub(r'-\s*\n\s*', '', sub_clean)
        sub_clean    = re.sub(r'\s*\n\s*', ' ', sub_clean)
        subject_line = re.sub(r'^"|"$', '', sub_clean).strip()

    if subject_line and email_body.startswith(subject_line):
        email_body = email_body[len(subject_line):].strip()

    email_body = email_body.strip()
    if not email_body:
        return "", "ERROR: Email body is completely empty after parsing."

    # CHECK 1: Word count
    word_count = len(email_body.split())
    if word_count < 50:
        return "", f"ERROR: Too short ({word_count} words) — genuinely incomplete."

    # CHECK 2: Both section headers must exist
    has_sf = bool(re.search(r'Salesforce services', email_body, re.IGNORECASE))
    has_ai = bool(re.search(r'AI services',         email_body, re.IGNORECASE))
    if not has_sf or not has_ai:
        missing = []
        if not has_sf: missing.append("Salesforce services—")
        if not has_ai: missing.append("AI services—")
        return "", f"ERROR: Missing section(s): {', '.join(missing)} — needs retry."

    # CHECK 3: Total bullets across full email — expect 8 (min 6)
    bullet_matches = re.findall(r'(?m)^[\s]*[\*•\-\–\—]|â€¢', email_body)
    if len(bullet_matches) < 6:
        return "", f"ERROR: Only {len(bullet_matches)} bullets — needs retry."
    if len(bullet_matches) < 8:
        logging.warning(f"⚠️  {len(bullet_matches)} bullets instead of 8 — saving anyway.")

    # CHECK 4: Ending punctuation on each bullet
    SENTENCE_END = ('.', '!', '?', '"', "'", ')')
    all_lines    = [l.strip() for l in email_body.split('\n') if l.strip()]
    bullet_lines = [l for l in all_lines if re.match(r'^[•*\-\–\—]', l)]

    for i, line in enumerate(bullet_lines, start=1):
        content = re.sub(r'^[•*\-\–\—]\s*', '', line).strip()
        words   = content.split()
        if len(words) >= 4 and not content.endswith(SENTENCE_END):
            return "", (
                f"ERROR: Bullet {i} cut mid-sentence — no dot at end. "
                f"Snippet: '...{content[-50:]}' — needs retry."
            )

    # CHECK 5: Last line of entire email must end properly
    last_line    = all_lines[-1] if all_lines else ""
    last_content = re.sub(r'^[•*\-\–\—]\s*', '', last_line).strip()
    if len(last_content.split()) >= 4 and not last_content.endswith(SENTENCE_END):
        return "", (
            f"ERROR: Email ends mid-sentence. "
            f"Last line: '...{last_content[-60:]}' — needs retry."
        )

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
 
        # safe_filename = (
        #     "".join(c for c in company_name if c.isalnum() or c in "._- ")
        #     .strip().replace(" ", "_").lower()
        # )
        safe_filename = _normalize_name(company_name)
 
        # json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        # pain_points_str = "Not available."
        # market_news     = "No recent market updates available."
 
        # if os.path.exists(json_path):
        #     with open(json_path, "r", encoding="utf-8") as f:
        #         research = json.load(f)
        #     if "pain_points" in research:
        #         pain_points_str = "\n".join([f"- {p}" for p in research["pain_points"]])
        #     if "recent_news" in research:
        #         market_news = "\n---\n".join([
        #             f"Title: {n.get('title')}\nSource: {n.get('source')}"
        #             for n in research["recent_news"][:3]
        #         ])

        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
        market_news     = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
 
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
        
        # SMART FALLBACK CHECK FOR EMPTY ARRAYS
        if not pain_points_str.strip():
            pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
        if not market_news.strip() or market_news == "\n---\n":
            market_news = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
 
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
    _retry_results = await asyncio.gather(*worker_coros, return_exceptions=True)
    for _r in _retry_results:
        if isinstance(_r, Exception):
            logging.error(f"❌ Retry worker crashed: {repr(_r)}")
 
    fixed = 0
    still_failed = []   # companies that failed even Cerebras+Groq retry
 
    for index, res in results.items():
        raw_email  = res.get("raw_email", "ERROR")
        source     = res.get("source", "Failed")
        cache_path = res.get("cache_path", "")
 
        # subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
        
        subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
        subject_line = _clean_email_text(subject_line)   # ← ADD
        email_body   = _clean_email_text(email_body)     # ← ADD
       
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
            parse_error = email_body if email_body.startswith("ERROR") else "Unknown parse failure"
            logging.warning(
                f"⚠️  [RETRY] {res.get('company', '?')} — Still failing.\n"
                f"    REASON  : {parse_error}\n"
                f"    RAW DUMP: {repr(raw_email[:1000])}"
            )
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
                        # safe_filename = (
                        #     "".join(c for c in company_name if c.isalnum() or c in "._- ")
                        #     .strip().replace(" ", "_").lower()
                        # )
                        safe_filename = _normalize_name(company_name)
                        # json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
                        # pain_points_str = "Not available."
                        # market_news     = "No recent market updates available."
                        # if os.path.exists(json_path):
                        #     with open(json_path, "r", encoding="utf-8") as f:
                        #         research = json.load(f)
                        #     if "pain_points" in research:
                        #         pain_points_str = "\n".join([f"- {p}" for p in research["pain_points"]])
                        #     if "recent_news" in research:
                        #         market_news = "\n---\n".join([
                        #             f"Title: {n.get('title')}\nSource: {n.get('source')}"
                        #             for n in research["recent_news"][:3]
                        #         ])

                        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
                        pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
                        market_news     = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
                        
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
                        
                        # SMART FALLBACK CHECK FOR EMPTY ARRAYS
                        if not pain_points_str.strip():
                            pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
                        if not market_news.strip() or market_news == "\n---\n":
                            market_news = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
                        prompt = _build_email_prompt(
                            company_name, industry, financial_intel,
                            market_news, pain_points_str, service_focus,
                        )
                    except Exception as rebuild_err:
                        logging.error(f"❌ [AZURE W{worker_num}] Could not rebuild prompt for {company}: {rebuild_err}")
                        azure_queue.task_done()
                        continue
 
                try:
                    _az_fut = asyncio.ensure_future(call_azure_async(prompt))
                    done, _ = await asyncio.wait({_az_fut}, timeout=45.0)
                    if not done:
                        _az_fut.cancel()
                        raise asyncio.TimeoutError()
                    raw_email = _az_fut.result()
                    raw_email = raw_email or "ERROR: Azure empty response"
                    # subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
                    
                    subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
                    subject_line = _clean_email_text(subject_line)   # ← ADD
                    email_body   = _clean_email_text(email_body)     # ← ADD
                    # if not subject_line or not email_body or "ERROR" in raw_email:
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
        
        # safe_filename = (
        #     "".join(c for c in company_name if c.isalnum() or c in "._- ")
        #     .strip()
        #     .replace(" ", "_")
        #     .lower()
        # )
        safe_filename = _normalize_name(company_name)
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
 
        # # Load research data
        # json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        # pain_points_str = "Not available."
        # market_news     = "No recent market updates available."
 
        # if os.path.exists(json_path):
        #     with open(json_path, "r", encoding="utf-8") as f:
        #         research = json.load(f)
        #     if "pain_points" in research:
        #         pain_points_str = "\n".join(
        #             [f"- {p}" for p in research["pain_points"]]
        #         )
        #     if "recent_news" in research:
        #         market_news = "\n---\n".join([
        #             f"Title: {n.get('title')}\nSource: {n.get('source')}"
        #             for n in research["recent_news"][:3]
        #         ])
        #         logging.info(f"📊 News loaded for {company_name}")
 
        # # UPDATED: Only read Annual Revenue and Total Funding
        # financial_intel = (
        #     f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
        #     f"Total Funding: {row.get('Total Funding', 'N/A')}"
        # )

        # UPDATED: Only read Annual Revenue and Total Funding (MOVED UP)
        financial_intel = (
            f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
            f"Total Funding: {row.get('Total Funding', 'N/A')}"
        )

        # Load research data
        json_path       = os.path.join(json_data_folder, f"{safe_filename}.json")
        pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
        market_news     = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
 
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
        
        # SMART FALLBACK CHECK FOR EMPTY ARRAYS LIKE MATRIX IT
        if not pain_points_str.strip():
            pain_points_str = f"Specific data not found. Please infer 2 highly critical business pain points for a {industry} company operating with {financial_intel}."
        if not market_news.strip() or market_news == "\n---\n":
            market_news = f"No recent news found. Use their financial status ({financial_intel}) or a massive recent trend in the {industry} sector as the opening signal."
 
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
    _main_results = await asyncio.gather(*worker_coros, return_exceptions=True)
    for _r in _main_results:
        if isinstance(_r, Exception):
            logging.error(f"❌ Main worker crashed: {repr(_r)}")
 
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
 
        # subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
        
        subject_line, email_body = _parse_email_output_combined(raw_email) if service_focus.lower() == "combined" else _parse_email_output(raw_email)
        subject_line = _clean_email_text(subject_line)   # ← ADD
        email_body   = _clean_email_text(email_body)     # ← ADD
        
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
 
def run_email_pipeline(
    df:                 pd.DataFrame,
    json_data_folder:   str  = "research_cache",
    service_focus:      str  = "salesforce",
    email_cache_folder: str  = "email_cache",
) -> pd.DataFrame:
    """
    Synchronous entry point — called from app1.py callback (runs in a thread).
    
    KEY FIX: Each call creates a FRESH event loop.
    - KeyWorker._get_lock() now detects loop mismatch and creates a fresh Lock
    - So stale Lock problem is gone even with fresh loops each time
    - asyncio.wait_for replaced with ensure_future+asyncio.wait (Python 3.14 safe)
    """
    # Always create a fresh loop per call.
    # KeyWorker._get_lock() handles Lock recreation automatically when loop changes.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _async_email_runner(df, json_data_folder, service_focus, email_cache_folder)
        )
    finally:
        try:
            loop.close()
        except Exception:
            pass

# Keep old name as alias so nothing breaks
run_serpapi_email_generation = run_email_pipeline


# ==============================================================================
# STANDALONE ENTRY POINT
# ==============================================================================
 
if __name__ == "__main__":
    logging.info("🚀 Running 3-API async pipeline (Gemini + Cerebras + Groq)…")
 
    CSV_FILE_PATH   = r"C:\Users\user\Desktop\Solution_Reverse_Enginnring\500_deployement - Copy\IT_Services_Filtered - Sheet9 (5).csv"
    TXT_OUTPUT_FILE = "Combined.txt"
    LOCAL_SERVICE_MODE = "combined"
 
    try:
        if os.path.exists(CSV_FILE_PATH):
 
            df = pd.read_csv(CSV_FILE_PATH)
            logging.info(f"Total rows in dataset: {len(df)}")
 
            # df["Industry"] = (
            #     df["Industry"]
            #     .fillna("")
            #     .astype(str)
            #     .str.strip()
            #     .str.lower()
            # )
 
            # filtered_df = df[df["Industry"] == "information technology & services"]
            # logging.info(f"IT Services companies found: {len(filtered_df)}")
 
            # if len(filtered_df) == 0:
            #     logging.error("❌ No IT Services companies found.")
            #     sys.exit(1)
 
            # test_df = filtered_df.sample(
            #     n=min(10, len(filtered_df)),
            #     random_state=None,
            # ).reset_index(drop=True)
            # logging.info(f"Selected {len(test_df)} companies for test run.")
            test_df = df.sample(
                n=min(10, len(df)),
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
