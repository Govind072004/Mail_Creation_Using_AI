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
# Import both legacy helpers AND new pool system
# from api_rotating_claude import (
#     # get_google_key, get_google_count,
#     # get_groq_key,   get_groq_count,
#     KeyWorker,      build_worker_pool,
#     get_azure_config,
# )
from api_rotating_claude import (
    KeyWorker,      build_worker_pool,
    get_azure_config,
)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/pipeline_run.log", mode="w", encoding="utf-8"),
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
_cb_lock: asyncio.Lock | None = None   # created lazily inside event loop


def _get_cb_lock() -> asyncio.Lock:
    global _cb_lock
    if _cb_lock is None:
        _cb_lock = asyncio.Lock()
    return _cb_lock


# ==============================================================================
# API CALL FUNCTIONS  — one per provider, no semaphores needed (KeyWorker handles timing)
# ==============================================================================

async def call_gemini_async(prompt: str, api_key: str) -> str:
    """
    Google Gemini 2.5 Flash
    Free tier: 10 RPM, ~500 RPD per key.
    Role: PRIMARY — highest quality, used first for every email.
    """
    client   = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text


async def call_cerebras_async(prompt: str, api_key: str) -> str:
    """
    Cerebras gpt-oss-120b  (OpenAI-compatible SDK)
    Free tier: 30 RPM, 64K TPM, 14,400 RPD — effectively unlimited daily.
    Role: SECONDARY — strong 120B model, takes over when Gemini keys are busy/exhausted.
    """
    client   = AsyncCerebras(api_key=api_key)
    response = await client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_completion_tokens=1500,
    )
    return response.choices[0].message.content


async def call_groq_async(prompt: str, api_key: str) -> str:
    """
    Groq llama-4-scout-17b-16e-instruct
    Free tier: 30 RPM, 30K TPM, TPD=500K → ~155 emails/day per key.
    Role: OVERFLOW — used when Gemini + Cerebras workers are all in their sleep window.

    NOTE: We use llama-4-scout (30K TPM) instead of llama-3.3-70b (12K TPM).
    This gives 2.5× better daily throughput per key at near-equivalent quality
    for structured, prompt-driven email generation.
    """
    client   = AsyncGroq(api_key=api_key)
    response = await client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content

async def call_azure_async(prompt: str) -> str:
    """
    Azure OpenAI GPT-4o Mini — EMERGENCY FALLBACK
    Paid API, no rate limits.
    Fires ONLY when retry_count >= 3
    (Gemini + Cerebras + Groq teeno fail kar chuke hain)
    """
    config = get_azure_config()

    client = AsyncAzureOpenAI(
        api_key        = config["api_key"],
        azure_endpoint = config["endpoint"],
        api_version    = config["api_version"],
    )

    response = await client.chat.completions.create(
        model       = config["deployment"],
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.1,
        max_tokens  = 1500,
    )
    return response.choices[0].message.content

# ==============================================================================
# PROMPT BUILDER  — extracted into its own function for clean separation
# The prompt text is preserved EXACTLY from the original — do not edit.
# ==============================================================================

def _build_email_prompt(
    company:    str,
    industry:   str,
    financials: str,
    market_news:str,
    pain_points:str,
    service_focus: str,
) -> str:
    """Returns the full email-generation prompt for a single company."""

    return f"""
============================================================
ROLE & CORE OBJECTIVE
============================================================
You are a senior B2B sales copywriter for **AnavClouds** specializing in highly personalized outbound emails for tech/consulting companies. Emails must feel human, personal, and manually written — never templated or marketing-heavy.

CORE RULE: Every element must position and sell {service_focus} only. No other services.
- Mention "AnavClouds" once, naturally, in the value proposition block only.

============================================================
DATA CONTEXT
============================================================
- Company: {company}
- Industry: {industry}
- Financial Intel: {financials}
- Market News: {market_news}
- Known Pain Points: {pain_points}
- Service Focus: {service_focus}

============================================================
PRE-WRITING ENGINES (THINK INTERNALLY — DO NOT OUTPUT)
============================================================

**A. SIGNAL EXTRACTION**
From market_news and financials, extract signals indicating: Growth, Change, Pressure, Complexity, Expansion, Hiring surge, or Technology shift.
Classify each as: Growth / Operational / Technology / Go-to-Market Signal.
Ignore generic PR language. Map internally as: Signal → Business Meaning.

**B. SERVICE FOCUS LOGIC (OVERRIDES ALL OTHER LOGIC)**
- SERVICE MODE = {service_focus}
- "salesforce" → Pitch ONLY Salesforce. Never mention AI, ML, automation, agents, LLM, or data platforms.
- "ai" → Pitch ONLY AI/ML/data/automation. Never mention Salesforce, CRM, Apex, LWC, Flows.
- "combined" → Blend Salesforce + AI naturally.

**C. PAIN INFERENCE**
Using signals, industry, size, and {pain_points} (prioritize these):
1. Infer likely operational bottlenecks, process inefficiencies, data/workflow gaps, scaling friction.
2. Do NOT invent pains. Base heavily on {pain_points}.
Map internally as: Primary Business Friction → Supporting Evidence.

**D. PAIN-TO-LINE MAPPING**
Step 1: Pick the STRONGEST pain from {pain_points}.
Step 2: Convert to an outcome phrase (not a problem statement).
Step 3: Use that outcome phrase naturally in body line 2.
Test: A reader must think "They clearly understand what we're dealing with."

**E. SOLUTION MAPPING**
Map friction to AnavClouds capabilities:
- Revenue leakage / sales ops / forecasting → Salesforce
- Process automation / data silos / intelligence → AI & Automation
- Multi-system complexity → Salesforce + AI blend
Frame as: outcomes, not service descriptions. Tone: Curious Peer, not Eager Vendor.

============================================================
SERVICE CAPABILITIES (Use only based on SERVICE MODE)
============================================================
If AI / Data / Automation-focused:
- Build enterprise AI agents using custom LLMs and RAG pipelines to automate insights from your company data.
- Develop predictive ML models for lead scoring, demand forecasting, and churn prediction to help your team prioritize high-value opportunities.
- Design scalable ETL pipelines and modern data warehouses for accurate, real-time analytics.
- Automate end-to-end business processes with Python and AI frameworks, freeing staff from repetitive tasks while maintaining alignment with existing workflows.
- Highlight certified AI, data science, ML, and automation experts who are screened, tested, and ready-to-work.
- Emphasize flexible hiring models (contract, contract-to-hire, remote staffing).
- Highlight fast onboarding in days.
- Stress technical match + culture-fit.

If Salesforce-focused:
- Implement Salesforce Sales, Service, Health, and Financial Services Clouds tailored to your company's workflows for unified operations.
- Develop custom Apex code, Lightning Web Components (LWC), and complex Flow automations to reduce manual effort and improve efficiency.
- Build or integrate AppExchange apps, automate marketing campaigns via Pardot/Marketing Cloud, and manage multi-org migrations to streamline cross-team processes.
- Provide 24/7 managed services including proactive system health checks to detect and fix technical debt before it impacts performance.
- Showcase certified Salesforce developers, admins, consultants across multiple clouds.
- Highlight flexible engagement models (contract, contract-to-hire, remote staffing).
- Emphasize quick onboarding.
- Stress skills match + culture-fit.

If both Salesforce + AI appear:
- Integrate predictive AI/ML models with Salesforce dashboards to forecast opportunities, optimize lead scoring, and drive data-backed decisions.
- Automate workflows combining Lightning Components, Flows, and Python-based AI scripts for seamless operations across CRM and analytics systems.
- Build enterprise AI agents that interact with Salesforce data to answer queries, detect anomalies, and provide actionable insights in real time.
- Ensure technical and cultural alignment with fast onboarding so the team can start contributing immediately while maintaining continuity with existing processes.
- Present certified professionals skilled in both.
- Highlight engagement flexibility.
- Emphasize fast onboarding.
- Stress dual technical + culture alignment.
============================================================
WRITING RULES (CRITICAL — ENFORCE ALL)
============================================================

**TONE & STYLE:**
- Conversational, natural, human. Sound like a busy professional writing quickly.
- Use contractions (don't, we're, it's). Allow slight rhythm irregularity.
- Short, simple, direct sentences. Vary sentence length — avoid predictable patterns.
- No brochure language. No robotic copy. No perfectly balanced structure.
- Simplify any sentence that sounds like marketing copy.

**FORBIDDEN PHRASES (SPAM TRIGGERS — NEVER USE):**
reach out, touch base, circle back, game-changer, cutting-edge, best-in-class, world-class, I wanted to connect, Hope this finds you well, Let me know if you're interested, Would love to, Excited to share, Thrilled to, Scale your business, Drive results, Unlock potential, Quick call, Quick chat, Hop on a call, Free consultation, No obligation, Risk-free, Revolutionize, Transform, Disrupt, I came across your profile, Just checking in, Here we are, team members, new

**FORBIDDEN PUNCTUATION:** No exclamation marks (!). No ALL CAPS for emphasis.

**BANNED WORDS (STRICTLY PROHIBITED IN SUBJECT + BODY):**
accelerate, certified, Offer, optimize, enhance, Success, great, Financial, fast, performance, new, Here, leverage, synergy, streamline, empower, solutions, deliverables, bandwidth, key hire, mission-critical, investment

**BANNED PATTERNS:**
- Any sentence starting with: "I wanted to"
- Any sentence ending with: "...let me know!"
Use synonyms/variations of all banned/forbidden words. If any appear, rewrite before output.

**FORMATTING RULES:**
- No bullet points in opening or positioning blocks — prose only.
- No feature lists. No structured sales formatting. No slogans or taglines.
- Email must read like an internal professional message, not marketing content.

**CTA RULE:** NO call-to-action. No ask for call/meeting/reply/conversation. No ending question. Email ends immediately after the final bullet.

**MICRO-PERSONALIZATION:** Reference one specific detail from financials or market_news. React as a peer — don't summarize. Comment briefly.

============================================================
SUBJECT LINE RULES
============================================================
**Structure:** [Desired Outcome] + without + [Core Friction]

**Process:**
1. Identify strongest operational friction (from Signal Extraction + Pain Inference).
2. Translate to human-understandable business pain (not technical).
3. Convert to simple outcome-focused phrase.
4. Combine: "[Outcome] without [Friction]"

**Rules:**
- No mention of AnavClouds, Salesforce, AI, automation, services, staffing, features, or tools.
- No buzzwords, hype, or marketing phrases.
- Must sound like a helpful internal peer note.
- Must clearly preview the email's topic.

Examples of structure (do NOT copy literally):
- Hire faster without the hiring chaos
- Better data visibility without operational complexity

============================================================
EMAIL STRUCTURE (STRICTLY FOLLOW)
============================================================

**SUBJECT:** [Outcome] without [Friction]

**Hi ,**

**Block 1 — Hyper-Personalized Opening (2 sentences MAX):**
- Sentence 1: Reference exactly ONE specific news item, product update, or milestone from the data. React like you noticed it this morning — don't summarize it.
- Sentence 2: Connect the observation to a natural business direction. No pain mention yet.
  **Rules**: Reference exactly ONE specific piece of news, product update, press release, LinkedIn post, or company milestone from the provided data.
- Do NOT summarize the news. React to it like a peer who noticed it.
- Make it feel like you read about them this morning, not like you researched them.
- Keep it tight. Two sentences. No more.
- The second sentence should connect that observation to a natural business direction — without mentioning pain.

Example logic (do NOT copy this literally):
  Sentence 1 → "I noticed [Company] recently [specific thing from news]."
  Sentence 2 → "That kind of move usually [natural business implication] — curious how the team is managing it at this stage."


**Block 2 — AnavClouds Positioning (exactly 2 lines):**
- Line 1: Introduce AnavClouds once, naturally. Frame what we do as the logical next layer for where they're heading. Mention 2–3 work areas in flowing prose — no bullets.
  Format logic: "At AnavClouds, we help companies like yours [outcome] by [method]" — rewritten naturally, non-templated.
- Line 2 (STRICT): State directly what AnavClouds helps fix, mapped to THIS company's exact pain points.
  PROCESS: (1) Read {pain_points} → (2) Pick 2 strongest pains → (3) Convert each to outcome → (4) Write ONE natural sentence combining both outcomes. Do NOT list them.
  Format: "We help teams [outcome of pain 1] and [outcome of pain 2]."

**Block 3 — Value Bridge & Bullets:**
Write ONE natural transition sentence (randomly chosen from list below) ending with ":"

Transition options (randomly pick one):
- Here are some ways we can help:
- Here's what usually helps in situations like this:
- A few practical ways teams simplify this:
- Here's what teams often find useful:
- A simple way to approach this:

[blank line MUST exist after the colon]

* Point 1 — Direct technical fix for strongest pain point
* Point 2 — Broader {industry} workflow, data setup, or tech debt improvement
* Point 3 — Direct fix for second strongest pain from {pain_points} — framed as outcome, not staffing
* Point 4 — One specific technical capability from {service_focus} that only a specialist can deliver — tied to their industry context

**BULLET FORMAT RULES (CRITICAL):**
- Exactly ONE space before the colon ":" on the transition line
- Exactly ONE blank line after the transition line before bullets begin
- Each bullet on its own line, separated by a blank line
- Use ONLY "*" for bullets
- Bullets must sound conversational, NOT marketing copy
- Avoid symmetry between bullets
- No bullet immediately after the colon

**No signature. Email ends after final bullet.**

============================================================
FINAL VALIDATION (BEFORE OUTPUT)
============================================================
1. Did you use "solutions", "streamline", or any banned/forbidden word? → Rewrite.
2. Does any sentence sound like marketing copy? → Simplify.
3. Is there a CTA, closing question, or sign-off? → Remove.
4. Output ends immediately after the final bullet.
5. Check anti-AI markers: No perfect symmetry, no over-polished structure, no corporate phrasing, natural rhythm throughout.

============================================================
STRICT OUTPUT FORMAT (DO NOT CHANGE)
============================================================
You must output exactly:

SUBJECT:
[Outcome-driven subject using: Desired Outcome + without + Core Friction]

Hi ,

[2 lines hyper-personalized company observation — based ONLY on
specific news or financial signal. NO industry name. NO "imagine".
NO generic sector statements.]

[2 lines showing AnavClouds understand & solve THEIR exact pain — no selling, no features, no services]

[here add one random transition line from the list above]:

• Bullet 1 — direct solution to their pain
• Bullet 2 — operational / infrastructure improvement
• Bullet 3 — direct solution to their second pain point
• Bullet 4 — specialist-level technical depth only a {service_focus} expert can deliver for {industry}
"""


# ==============================================================================
# WORKER COROUTINE — one instance per KeyWorker (18 total: 9G + 3C + 6Q)
# ==============================================================================

async def _email_worker_loop(
    worker_id:      int,
    key_worker:     KeyWorker,
    queue:          asyncio.Queue,
    results:        dict,
    total_expected: int,
    email_cache_folder: str,
    service_focus:  str,
    worker_pool:    list,   # ← ADD THIS

) -> None:
    """
    Persistent worker coroutine bound to ONE API key.

    Flow per task:
      1. Pull task from shared queue  (blocks up to 5s, then checks if done)
      2. Call wait_and_acquire()      (sleeps until RPM window is open)
      3. Fire the API call
      4. On 429        → mark_429() on key, requeue task, continue loop
      5. On daily limit → mark_daily_exhausted(), requeue task, EXIT loop
      6. On success    → parse + cache + write to results dict
      7. On hard error → increment circuit breaker, mark failed, continue
    """
    global CONSECUTIVE_FAILURES

    provider_label = key_worker.provider.capitalize()

    while True:

        # ── Exit condition: all emails collected ─────────────────────
        if len(results) >= total_expected:
            break

        # ── Pull next task (5 s timeout) ─────────────────────────────
        try:
            task = await asyncio.wait_for(queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            # Queue temporarily empty — check if truly done
            if len(results) >= total_expected:
                break
            continue

        company     = task["company"]
        index       = task["index"]
        full_prompt = task["prompt"]
        cache_path  = task["cache_path"]
        retry_count = task.get("retry_count", 0)

        # # ── Hard retry cap — prevent infinite loops ───────────────────
        # if retry_count > 4:
        #     logging.error(f"❌ [{worker_id}] Max retries for {company}. Marking failed.")
        #     results[index] = {
        #         "company":   company,
        #         "source":    "Failed",
        #         "raw_email": "ERROR: Max retries exceeded across all APIs.",
        #     }
        #     queue.task_done()
        #     continue

        # ── Attempt 3 → Azure Emergency Fallback ─────────────────────
        if retry_count >= 3:
            logging.warning(
                f"🔴 [W{worker_id:02d}|AZURE] {company} → "
                f"3 attempts fail. send to Azure."
            )
            try:
                raw_email = await call_azure_async(full_prompt)
                raw_email = raw_email or "ERROR: Azure empty response"

                logging.info(
                    f"✅ [W{worker_id:02d}|AZURE] {company} — Done via Azure."
                )
                results[index] = {
                    "company":    company,
                    "source":     "Azure",
                    "raw_email":  raw_email,
                    "cache_path": cache_path,
                }

            except Exception as azure_exc:
                logging.error(
                    f"❌ [W{worker_id:02d}|AZURE] Azure is failed: {azure_exc}"
                )
                results[index] = {
                    "company":    company,
                    "source":     "Failed",
                    "raw_email":  f"ERROR: Azure failed — {azure_exc}",
                    "cache_path": cache_path,
                }

            queue.task_done()
            continue

        
        # if key_worker.provider == "groq":
        #     gemini_or_cerebras_free = any(
        #         not w.is_exhausted and not w.is_cooling
        #         for w in worker_pool
        #         if w.provider in ("gemini", "cerebras")
        #     )
        #     if gemini_or_cerebras_free:
        #         await queue.put(task)
        #         queue.task_done()
        #         await asyncio.sleep(1.5)
        #         continue
       
        ready = await key_worker.wait_and_acquire()
        if not ready:
            # Daily quota for this key is done — requeue and exit this worker
            logging.warning(
                f"⚠️  Worker {worker_id} ({provider_label}) daily quota reached. "
                f"Requeueing {company}."
            )
            task["retry_count"] = retry_count  # don't penalise the task
            await queue.put(task)
            queue.task_done()
            break  # this worker is done for the day

        # ── Fire the API call ─────────────────────────────────────────
        logging.info(
            f"[W{worker_id:02d}|{provider_label}] → {company} "
            f"(attempt {retry_count + 1})"
        )

        try:
            if key_worker.provider == "gemini":
                raw_email = await call_gemini_async(full_prompt, key_worker.api_key)
            elif key_worker.provider == "cerebras":
                raw_email = await call_cerebras_async(full_prompt, key_worker.api_key)
            else:
                raw_email = await call_groq_async(full_prompt, key_worker.api_key)
            raw_email = raw_email or "ERROR: API returned empty response"

            # ── SUCCESS ──────────────────────────────────────────────
            # async with _get_cb_lock():
            #     CONSECUTIVE_FAILURES = 0
            async with _get_cb_lock():
                CONSECUTIVE_FAILURES = 0

            key_worker.reset_retry_count()  

            logging.info(f"✅ [W{worker_id:02d}|{provider_label}] {company} — Done.")

            results[index] = {
                "company":   company,
                "source":    provider_label,
                "raw_email": raw_email,
                "cache_path": cache_path,
            }
            queue.task_done()

        except Exception as exc:
            err_lower = str(exc).lower()

            # ── 429 / Rate-limit ─────────────────────────────────────
            if any(
                kw in err_lower
                for kw in ["429", "rate_limit", "rate limit", "quota_exceeded",
                            "resource_exhausted", "too many requests"]
            ):
                logging.warning(
                    f"⚠️  [W{worker_id:02d}|{provider_label}] 429 on {company}. "
                    f"Cooling 65s, requeueing (attempt {retry_count + 1})."
                )
                key_worker.mark_429()
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()

            # ── Daily / Monthly limit ────────────────────────────────
            elif any(
                kw in err_lower
                for kw in ["daily", "exceeded your daily", "monthly", "billing"]
            ):
                logging.warning(
                    f"⚠️  [W{worker_id:02d}|{provider_label}] Daily limit hit. "
                    f"Exhausting key, requeueing {company}."
                )
                key_worker.mark_daily_exhausted()
                task["retry_count"] = retry_count + 1
                await queue.put(task)
                queue.task_done()
                break  # this key is done for the day

            # ── Unknown hard error ───────────────────────────────────
            else:
                logging.error(
                    f"❌ [W{worker_id:02d}|{provider_label}] Hard error on "
                    f"{company}: {exc}"
                )
                async with _get_cb_lock():
                    CONSECUTIVE_FAILURES += 1
                    if CONSECUTIVE_FAILURES >= MAX_FAILURES:
                        logging.critical(
                            "🛑 CIRCUIT BREAKER TRIPPED — "
                            f"{CONSECUTIVE_FAILURES} consecutive failures. "
                            "Halting pipeline to protect API keys."
                        )
                        # sys.exit(1)
                        raise RuntimeError("Circuit breaker: too many consecutive failures.")

                results[index] = {
                    "company":    company,
                    "source":     "Failed",
                    "raw_email":  f"ERROR: {exc}",
                    "cache_path": cache_path,
                }
                queue.task_done()


# ==============================================================================
# RESULT PARSER  — extracts subject + body from raw model output
# ==============================================================================

# def _parse_email_output(raw_email: str) -> tuple[str, str]:
#     """
#     Returns (subject_line, email_body) extracted from the model's raw output.
#     Mirrors the exact regex logic from the original file.
#     """
#     subject_line = ""
#     email_body   = raw_email

# #     if not raw_email.startswith("ERROR"):
# def _parse_email_output(raw_email: str) -> tuple[str, str]:
#     # None check — Cerebras/Groq kabhi kabhi empty return karte hain
#     if not raw_email:
#         return "", "ERROR: API returned empty response"
    

#     subject_line = ""
#     email_body   = raw_email

#     if not raw_email.startswith("ERROR"):
#         sub_match = re.search(r'(?i)\*?\*?SUBJECT:\*?\*?\s*(.+)', raw_email)
#         if sub_match:
#             subject_line = sub_match.group(1).strip()

#         body_match = re.search(r'(?i)(Hi\s*,?.*)', raw_email, re.DOTALL)
#         if body_match:
#             email_body = body_match.group(1).strip()
#         else:
#             email_body = re.sub(
#                 r'(?i)\*?\*?SUBJECT:\*?\*?\s*.+\n', '', raw_email
#             ).strip()

#     return subject_line, email_body

# def _parse_email_output(raw_email: str) -> tuple[str, str]:
#     # None / empty check
#     if not raw_email:
#         return "", "ERROR: API returned empty response"

#     subject_line = ""
#     email_body   = raw_email

#     if not raw_email.startswith("ERROR"):

#         # ── Step 1: Extract Subject ───────────────────────────────
#         sub_match = re.search(r'(?i)\*?\*?SUBJECT:\*?\*?\s*(.+)', raw_email)
#         if sub_match:
#             subject_line = sub_match.group(1).strip()

#         # ── Step 2: Find "Hi" only at LINE START (re.MULTILINE) ──
#         # Fix: (?m) means ^ = start of any line, not just string start
#         # This prevents matching "Hi" inside words like "diminisHing"
#         body_match = re.search(r'(?m)^(Hi[\s,].+)', raw_email, re.DOTALL)
#         if body_match:
#             email_body = body_match.group(1).strip()
#         else:
#             # Fallback: remove subject line and return rest
#             email_body = re.sub(
#                 r'(?i)\*?\*?SUBJECT:\*?\*?\s*.+\n', '', raw_email
#             ).strip()

#         # ── Step 3: Remove subject if it leaked into body top ─────
#         # Handles case where subject appears above "Hi" in body
#         if subject_line and email_body.startswith(subject_line):
#             email_body = email_body[len(subject_line):].strip()

#     return subject_line, email_body



# def _parse_email_output(raw_email: str) -> tuple[str, str]:
#     if not raw_email:
#         return "", "ERROR: API returned empty response"

#     # 1. Strip rogue markdown (e.g., ```text, ```html)
#     clean_text = raw_email.strip()
#     clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
#     clean_text = re.sub(r'\n```$', '', clean_text)
#     clean_text = clean_text.strip()

#     if clean_text.startswith("ERROR"):
#         return "", clean_text

#     subject_line = ""
#     email_body = clean_text
#     pre_body = ""

#     # 2. Layer 1: Look for ANY standard greeting, not just "Hi"
#     # Matches: "Hi ,", "Hello ,", "Hey,", "Dear John,"
#     body_match = re.search(r'(?m)^((?:Hi|Hello|Hey|Dear)[\s,].*)', clean_text, re.DOTALL | re.IGNORECASE)

#     if body_match:
#         email_body = body_match.group(1).strip()
#         pre_body = clean_text[:body_match.start()].strip()
#     else:
#         # 3. Layer 2: Fallback if greeting is completely missing
#         # Split by the first double-newline. The top block is the subject.
#         parts = re.split(r'\n{2,}', clean_text, maxsplit=1)
#         if len(parts) == 2:
#             pre_body, email_body = parts[0].strip(), parts[1].strip()
#         else:
#             # Absolute worst-case: No greeting, no double newlines
#             pre_body = ""
#             email_body = clean_text

#     # 4. Clean the Subject Block
#     if pre_body:
#         # Remove "SUBJECT:" and any bold formatting markers
#         sub_clean = re.sub(r'(?i)\*?\*?SUBJECT:\*?\*?\s*', '', pre_body).strip()
        
#         # Re-merge words broken by accidental line wraps (Fixes the "hifts" issue)
#         sub_clean = re.sub(r'-\s*\n\s*', '', sub_clean)  # Fixes hyphenated breaks
#         sub_clean = re.sub(r'\s*\n\s*', ' ', sub_clean)  # Fixes normal line breaks
        
#         # Remove trailing/leading quotes
#         subject_line = re.sub(r'^"|"$', '', sub_clean).strip()

#     # 5. Last safety check: If subject still somehow leaked into body top
#     if subject_line and email_body.startswith(subject_line):
#         email_body = email_body[len(subject_line):].strip()

#     return subject_line, email_body


def _parse_email_output(raw_email: str) -> tuple[str, str]:
    if not raw_email:
        return "", "ERROR: API returned empty response"

    # 1. Clean Markdown wrappers
    clean_text = raw_email.strip()
    clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
    clean_text = re.sub(r'\n```$', '', clean_text)
    clean_text = clean_text.strip()

    if clean_text.startswith("ERROR"):
        return "", clean_text

    subject_line = ""
    email_body = clean_text
    pre_body = ""

    # 2. Extract Body using Reverse Search
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

    # 3. Clean Subject Block
    if pre_body:
        sub_clean = re.sub(r'(?i)\*?\*?SUBJECT:\*?\*?\s*', '', pre_body).strip()
        sub_clean = re.sub(r'-\s*\n\s*', '', sub_clean)
        sub_clean = re.sub(r'\s*\n\s*', ' ', sub_clean)
        subject_line = re.sub(r'^"|"$', '', sub_clean).strip()

    if subject_line and email_body.startswith(subject_line):
        email_body = email_body[len(subject_line):].strip()

    # =========================================================
    # 🚨 PRODUCTION-GRADE VALIDATION LAYER (GPT-Recommended) 🚨
    # =========================================================
    
    email_body = email_body.strip()
    if not email_body:
        return "", "ERROR: Email body is completely empty after parsing."

    # A. Word Count Check (Detects large but truncated emails)
    word_count = len(email_body.split())
    if word_count < 40:
        return "", f"ERROR: Truncated email (Only {word_count} words). Fails word count check."

    # B. Strict Bullet Detection (Handles *, •, -, and encoding corruptions like â€¢)
    bullet_matches = re.findall(r'(?m)^[\s]*[\*•\-\–\—]|â€¢', email_body)
    if len(bullet_matches) < 4:
        return "", f"ERROR: Incomplete generation. Found only {len(bullet_matches)} bullets, expected 4."

    # C. Abrupt Ending Check (No valid punctuation at the end)
    if email_body[-1] not in ['.', '!', '?', '"', '\'']:
        return "", "ERROR: Email body cut off mid-sentence (No ending punctuation)."

    # D. Half-Bullet Cutoff Check (Last line is too short / missing context)
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

    # Sirf Cerebras + Gemini workers
    # retry_workers = [
    #     w for w in worker_pool
    #     if w.provider in ("cerebras", "gemini")
    # ]
    retry_workers = [
    w for w in worker_pool
    if w.provider in ("cerebras", "groq")  # ← Groq added
]

    # ERROR wali rows find karo
    error_mask = (
        df_output["Generated_Email_Body"].astype(str).str.contains("ERROR", na=False) |
        df_output["Generated_Email_Subject"].isna() |
        (df_output["Generated_Email_Subject"].astype(str).str.strip() == "")
    )
    failed_indices = df_output[error_mask].index.tolist()

    if not failed_indices:
        logging.info("✅ No failed emails — skipping retry.")
        return df_output

    logging.info(f"\n🔁 AUTO-RETRY START — {len(failed_indices)} failed emails (Cerebras+Gemini only)\n")

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

        # Pain points + news — structured_company_data se
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

    # Retry workers launch karo
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

    # Results apply
    fixed = 0
    deleted_indices = []

    for index, res in results.items():
        raw_email  = res.get("raw_email", "ERROR")
        source     = res.get("source", "Failed")
        cache_path = res.get("cache_path", "")

        subject_line, email_body = _parse_email_output(raw_email)

        if subject_line and email_body and "ERROR" not in raw_email:
            # ✅ Fix ho gaya — update karo
            df_output.at[index, "Generated_Email_Subject"] = subject_line
            df_output.at[index, "Generated_Email_Body"]    = email_body
            df_output.at[index, "AI_Source"]               = f"{source}(retry)"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"subject": subject_line, "body": email_body, "source": source},
                    f, indent=4,
                )
            fixed += 1
        else:
            # ❌ Phir bhi fail — delete
            deleted_indices.append(index)

    # Failed rows delete karo
    if deleted_indices:
        df_output.drop(index=deleted_indices, inplace=True)
        df_output.reset_index(drop=True, inplace=True)

    logging.info(
        f"\n🔁 RETRY COMPLETE\n"
        f"   Fixed   : {fixed}\n"
        f"   Deleted : {len(deleted_indices)}\n"
    )
    return df_output

# ==============================================================================
# ASYNC RUNNER  — unified queue engine, replaces the old batch-loop
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

    os.makedirs(email_cache_folder, exist_ok=True)

    df_output = df.copy()
    df_output["Generated_Email_Subject"] = ""
    df_output["Generated_Email_Body"]    = ""
    df_output["AI_Source"]               = ""

    # ── Build worker pool ─────────────────────────────────────────
    try:
        worker_pool = build_worker_pool()
    except RuntimeError as e:
        logging.critical(str(e))
        raise

    # ── Build task queue + handle cache hits ──────────────────────
    queue          = asyncio.Queue()
    tasks_to_run   = []

    for index, row in df_output.iterrows():
        company_name  = str(row.get("Company Name", "")).strip()
        industry      = str(row.get("Industry", "Technology"))
        safe_filename = (
            "".join(c for c in company_name if c.isalnum() or c in "._- ")
            .strip()
            .replace(" ", "_")
            .lower()
        )
        cache_path = os.path.join(
            email_cache_folder, f"{safe_filename}_{service_focus.lower()}.json"
        )

        # ── Cache check ───────────────────────────────────────────
        if os.path.exists(cache_path):
            logging.info(f"⏩ Cache hit: {company_name}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            df_output.at[index, "Generated_Email_Subject"] = cached.get("subject", "")
            df_output.at[index, "Generated_Email_Body"]    = cached.get("body",    "")
            df_output.at[index, "AI_Source"]               = cached.get("source",  "Cache")
            continue

        # ── Load research data ────────────────────────────────────
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

        financial_intel = (
            f"Revenue: {row.get('Annual Revenue', 'N/A')}, "
            f"Total Funding: {row.get('Total Funding', 'N/A')}"
        )

        # Log prompt inputs for debugging
        logging.info(
            f"\n{'='*48}\n"
            f"PROMPT INPUT\n"
            f"  Company    : {company_name}\n"
            f"  Industry   : {industry}\n"
            f"  Financials : {financial_intel}\n"
            f"  News lines : {len(market_news.splitlines())}\n"
            f"  Pains      : {len(pain_points_str.splitlines())}\n"
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

    # ── Nothing to do? ────────────────────────────────────────────
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

    # ── Fill queue ────────────────────────────────────────────────
    for task in tasks_to_run:
        await queue.put(task)

    # ── Shared results dict ───────────────────────────────────────
    results: dict = {}

    # ── Launch all 18 worker coroutines concurrently ──────────────
    worker_coros = [
        _email_worker_loop(
            worker_id=i,
            key_worker=w,
            queue=queue,
            results=results,
            total_expected=total_expected,
            email_cache_folder=email_cache_folder,
            service_focus=service_focus,
            worker_pool=worker_pool,  # ← PASS THE POOL TO EACH WORKER
        )
        for i, w in enumerate(worker_pool)
    ]
    await asyncio.gather(*worker_coros, return_exceptions=True)

     

    # ── Parse results + write to DataFrame + save cache ──────────
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

        # Write cache only on clean success
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
    # ── Final stats ───────────────────────────────────────────────
    source_counts: dict = {}
    for res in results.values():
        s = res.get("source", "Unknown")
        source_counts[s] = source_counts.get(s, 0) + 1

    # logging.info(
    #     f"\n{'='*48}\n"
    #     f"PIPELINE COMPLETE\n"
    #     f"  Total processed : {total_expected}\n"
    #     f"  Success         : {success_count}\n"
    #     f"  Failed          : {fail_count}\n"
    #     f"  By source       : {source_counts}\n"
    #     f"{'='*48}\n"
    # )

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
# SYNCHRONOUS WRAPPER  — keeps full Streamlit compatibility
# ==============================================================================

def run_serpapi_email_generation(
    df:                 pd.DataFrame,
    json_data_folder:   str  = "structured_company_data",
    service_focus:      str  = "salesforce",
    email_cache_folder: str  = "/tmp/email_generation_cache_serpapi",
) -> pd.DataFrame:
    """
    Synchronous entry point.
    Safe to call from Streamlit, Jupyter, or any non-async context.
    """
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
# STANDALONE ENTRY POINT  — for direct CLI testing
# ==============================================================================

if __name__ == "__main__":
    logging.info("🚀 Running 3-API async pipeline (Gemini + Cerebras + Groq)…")

    CSV_FILE_PATH   = r"C:\Users\user\Desktop\Reverse Engineering\apollo-contacts-export (19) (2).csv"
    TXT_OUTPUT_FILE = "standalone_generated_email_IT_gemini_serp6.txt"
    LOCAL_SERVICE_MODE = "AI"

    try:
        if os.path.exists(CSV_FILE_PATH):

            # ── STEP 1: Load CSV ───────────────────────────────────
            df = pd.read_csv(CSV_FILE_PATH)
            logging.info(f"Total rows in dataset: {len(df)}")

            # ── STEP 2: Clean Industry column ─────────────────────
            df["Industry"] = (
                df["Industry"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            # ── STEP 3: Filter IT Services ────────────────────────
            filtered_df = df[df["Industry"] == "information technology & services"]
            logging.info(f"IT Services companies found: {len(filtered_df)}")

            # ── STEP 4: Sample companies ──────────────────────────
            if len(filtered_df) == 0:
                logging.error("❌ No IT Services companies found.")
                sys.exit(1)

            test_df = filtered_df.sample(
                n=min(10, len(filtered_df)),
                random_state=None,
            ).reset_index(drop=True)
            logging.info(f"Selected {len(test_df)} companies for test run.")

            # ── STEP 5: Run pipeline ───────────────────────────────
            result_df = run_serpapi_email_generation(
                test_df, service_focus=LOCAL_SERVICE_MODE
            )

            # ── STEP 6: Save to TXT ────────────────────────────────
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
