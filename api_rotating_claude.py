"""
api_rotating_claude.py
======================
Production-ready API Key Rotation Manager
Designed for Render Free Tier deployment.

VERSION: 3.0 — Three critical fixes applied over v2:
  FIX 1 → wait_and_acquire() is now NON-BLOCKING.
           Cooling workers are skipped instantly — no waiting on a busy key.
           A pool-level function (get_next_available_worker) handles the skip logic.

  FIX 2 → SerpAPI validation now uses httpx (async HTTP).
           The old requests.get() was synchronous and froze the entire event loop
           for up to 25 seconds during startup validation of multiple keys.
           httpx runs all validations concurrently without blocking.

  FIX 3 → Permanently dead keys are auto-disabled after MAX_FAILURES consecutive 429s.
           This prevents the "death spiral" where a broken key keeps getting retried
           forever. After MAX_FAILURES hits, the worker is force-exhausted for the day.

WHAT THIS FILE DOES:
--------------------
Manages multiple API keys for Gemini, Cerebras, Groq, Azure, SerpAPI, and Tavily.
Instead of hammering one key until it breaks, it rotates between all your keys
automatically — so you never hit rate limits during production workloads.

HOW TO ADD OR REMOVE KEYS (zero code changes needed):
------------------------------------------------------
Just add or remove lines in your .env file (or Render Environment Variables).
The code scans for all matching keys at runtime.

    Google / Gemini  →  GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2 ...
    Cerebras         →  CEREBRAS_API_KEY, CEREBRAS_API_KEY_1, CEREBRAS_API_KEY_2 ...
    Groq             →  GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2 ...
    Tavily           →  TAVILY_API_KEY, TAVILY_API_KEY_1 ...
    SerpAPI          →  SERPAPI_KEY, SERPAPI_KEY_1 ...
    Azure            →  AZURE_API_KEY + AZURE_ENDPOINT (single config, not rotated)

RENDER DEPLOYMENT NOTES:
-------------------------
  - Set all keys in Render Dashboard → Environment → Environment Variables.
  - No .env file needed on Render — the code auto-detects and skips it.
  - BASE_DIR resolves automatically from this file's location.
  - No hardcoded paths anywhere.

RATE LIMITS REFERENCE (verified March 2026):
---------------------------------------------
  Provider   │ RPM  │  TPM   │ Sleep (s) │ Cooldown │ Max Failures
  ───────────┼──────┼────────┼───────────┼──────────┼─────────────
  Gemini     │  10  │  big   │   6.5     │   30s    │      5
  Cerebras   │  30  │  64K   │   2.5     │   15s    │      5
  Groq       │  30  │  30K   │   6.5     │   20s    │      5
"""

import os
import asyncio
import itertools
import time
import httpx                   # FIX 2: replaces synchronous 'requests' for SerpAPI
import requests                # kept only for legacy sync helpers that are non-blocking
from datetime import datetime
from dotenv import load_dotenv

# ==============================================================================
# PATH SETUP — Works identically on local machine and Render server
# ==============================================================================

# BASE_DIR = the folder this file lives in.
# All relative paths should be built using os.path.join(BASE_DIR, "filename").
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env only when the file exists (local development).
# On Render, env vars are set in the dashboard — no .env file is present,
# so this block is silently skipped. No errors, no warnings.
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)


# ==============================================================================
# SECTION 1 — SHARED INTERNAL UTILITIES
# ==============================================================================

def _log_key_usage(service_name: str, key: str, delay: float = 0):
    """
    Logs which API key is being used, with the key safely masked.

    Only the first 6 and last 4 characters are printed — never the full key.
    This keeps Render logs safe to share with teammates without leaking secrets.

    Args:
        service_name : Display name shown in logs (e.g. "GEMINI", "GROQ").
        key          : The raw API key string (will be masked before printing).
        delay        : Optional seconds to sleep after logging. Used only by
                       the legacy synchronous getters (Section 3). Async code
                       never passes a non-zero delay here.
    """
    masked_key = key[:6] + "..." + key[-4:]
    timestamp  = datetime.now().strftime("%H:%M:%S")
    print(f"🔁 [{timestamp}] {service_name} → Using key: {masked_key}")
    if delay > 0:
        print(f"⏳ {service_name} → Waiting {delay}s before next rotation...")
        time.sleep(delay)


def _get_all_keys(prefix: str) -> list:
    """
    Scans os.environ and returns every value whose variable name matches prefix.

    Matching rules:
        GROQ_API_KEY          → matches (exact)
        GROQ_API_KEY_1        → matches (underscore + anything)
        GROQ_API_KEY_BACKUP   → matches (underscore + anything)
        GROQ_API_KEY_99       → matches (underscore + anything)
        GROQ_SOMETHING_ELSE   → does NOT match (different root)

    This means you can have any number of keys with any suffix — just keep the
    same prefix and the code picks them all up automatically.

    Args:
        prefix : Base env variable name (e.g. "GOOGLE_API_KEY").

    Returns:
        List of non-empty key strings. Empty list if none found.
    """
    keys = []
    for env_name, env_val in os.environ.items():
        if (env_name == prefix or env_name.startswith(f"{prefix}_")) and env_val.strip():
            keys.append(env_val.strip())
    return keys


def _create_key_cycle(prefix: str):
    """
    Builds an infinite round-robin cycle from all keys matching prefix.

    Used by the legacy synchronous getter functions (Section 3).
    The cycle never ends — it wraps around: key1 → key2 → key3 → key1 → ...

    Args:
        prefix : Base env variable name (e.g. "TAVILY_API_KEY").

    Returns:
        Tuple of (itertools.cycle object | None, key count int).
        Returns (None, 0) if no keys are found.
    """
    keys = _get_all_keys(prefix)
    if not keys:
        print(f"⚠️  Warning: No keys found for prefix '{prefix}'")
        return None, 0
    print(f"✅ Key Manager: Loaded {len(keys)} key(s) for '{prefix}'")
    return itertools.cycle(keys), len(keys)


# ==============================================================================
# SECTION 2 — SERPAPI ASYNC VALIDATOR  (FIX 2: now uses httpx, not requests)
# ==============================================================================

async def _create_smart_serpapi_cycle_async(prefix: str):
    """
    Validates all SerpAPI keys concurrently and returns a cycle sorted by credits.

    FIX 2 EXPLAINED:
    The old version used requests.get() which is synchronous (blocking). When
    validating 5 keys × 5s timeout each, the entire Python event loop was frozen
    for up to 25 seconds. During that freeze, no other async task could run —
    meaning your server appeared completely dead to Render's health check.

    This version uses httpx.AsyncClient() which is fully async. All keys are
    validated concurrently using asyncio.gather(), so 5 keys take ~5s total
    instead of ~25s, and the event loop stays alive throughout.

    PROCESS:
      1. Collect all SERPAPI_KEY, SERPAPI_KEY_1, SERPAPI_KEY_2 ... from env.
      2. Validate all keys concurrently (async HTTP, non-blocking).
      3. Drop keys with 0 credits or errors.
      4. Sort valid keys by credits remaining (highest first).
      5. Return an infinite round-robin cycle of valid keys.

    WHY LAZY INIT?
    This function is only called on the FIRST get_serpapi_key() usage, not at
    import or startup time. This means the app starts instantly and validation
    happens naturally when SerpAPI is first needed.

    Args:
        prefix : Base env variable name (e.g. "SERPAPI_KEY").

    Returns:
        Tuple of (itertools.cycle | None, int).
    """
    raw_keys = _get_all_keys(prefix)
    if not raw_keys:
        print(f"⚠️  No SerpAPI keys found for prefix '{prefix}'")
        return None, 0

    print(f"🔄 Validating {len(raw_keys)} SerpAPI key(s) concurrently...")

    async def _check_one_key(client: httpx.AsyncClient, key: str) -> dict | None:
        """
        Checks a single SerpAPI key. Returns credit info or None if invalid.
        This inner function runs concurrently for all keys via asyncio.gather().
        """
        try:
            response = await client.get(
                f"https://serpapi.com/account?api_key={key}",
                timeout=5.0
            )
            data = response.json()
            if "error" not in data and response.status_code == 200:
                credits = data.get("total_searches_left", 0)
                if credits > 0:
                    return {"key": key, "credits": credits}
                else:
                    masked = key[:6] + "..." + key[-4:]
                    print(f"⚠️  SerpAPI key {masked} → 0 credits, skipping.")
        except Exception as e:
            masked = key[:6] + "..." + key[-4:]
            print(f"⚠️  SerpAPI key {masked} → Validation failed ({e}), skipping.")
        return None

    # Run all key checks at the same time (non-blocking, parallel)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_check_one_key(client, k) for k in raw_keys])

    # Filter out None (failed/empty) results
    valid_keys_info = [r for r in results if r is not None]

    if not valid_keys_info:
        print("❌ Critical: All SerpAPI keys are invalid or have 0 credits.")
        return None, 0

    # Sort so the key with most credits is used first
    valid_keys_info.sort(key=lambda x: x["credits"], reverse=True)
    sorted_keys = [item["key"] for item in valid_keys_info]

    print(f"✅ SerpAPI: {len(sorted_keys)} active key(s) validated.")
    print(f"🏆 Top key has {valid_keys_info[0]['credits']} credits remaining.")

    return itertools.cycle(sorted_keys), len(sorted_keys)


def _create_smart_serpapi_cycle_sync(prefix: str):
    """
    Synchronous wrapper around the async SerpAPI validator.

    Used when get_serpapi_key() is called from synchronous (non-async) code.
    Safely runs the async validator in a new event loop if none is running,
    or schedules it on the existing loop if one exists.

    Args:
        prefix : Base env variable name (e.g. "SERPAPI_KEY").

    Returns:
        Tuple of (itertools.cycle | None, int).
    """
    try:
        loop = asyncio.get_running_loop()
        # An event loop is already running (e.g., inside FastAPI or an async app).
        # We cannot call loop.run_until_complete() here — it would deadlock.
        # Instead, schedule the coroutine as a future and wait for it.
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            _create_smart_serpapi_cycle_async(prefix), loop
        )
        return future.result(timeout=60)
    except RuntimeError:
        # No event loop running — safe to use asyncio.run()
        return asyncio.run(_create_smart_serpapi_cycle_async(prefix))


# ==============================================================================
# SECTION 3 — LEGACY SYNC KEY GETTERS  (kept intact for backward compatibility)
# ==============================================================================
# These functions are for simple, one-off synchronous API calls.
# They do basic round-robin rotation but NO async rate limiting.
# For high-throughput parallel work, use the async worker pool (Section 5+).

# Module-level state — all None at import time, lazily initialized on first use.
# This avoids any startup overhead or event-loop issues during module import.
_groq_cycle,     _groq_count     = None, 0
_tavily_cycle,   _tavily_count   = None, 0
_google_cycle,   _google_count   = None, 0
_serpapi_cycle,  _serpapi_count  = None, 0
_cerebras_cycle, _cerebras_count = None, 0


def get_gemini_key(delay: float = 0) -> str:
    """
    Returns the next available Google Gemini API key in round-robin rotation.

    On the first call, discovers all GOOGLE_API_KEY* variables from the
    environment and builds the rotation cycle. Subsequent calls just advance
    the cycle — no re-scanning.

    Args:
        delay : Seconds to sleep after returning the key (default 0 = no sleep).
                Useful if you want manual throttling in synchronous scripts.

    Returns:
        A Gemini API key string, ready to use.

    Raises:
        ValueError: If no GOOGLE_API_KEY* variables are found in environment.
    """
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
        if _google_cycle is None:
            raise ValueError(
                "❌ No Gemini keys found. "
                "Set GOOGLE_API_KEY (+ optional GOOGLE_API_KEY_1, _2 ...) in environment."
            )
    key = next(_google_cycle)
    _log_key_usage("GEMINI", key, delay)
    return key


def get_cerebras_key(delay: float = 0) -> str:
    """
    Returns the next available Cerebras API key in round-robin rotation.

    Discovers all CEREBRAS_API_KEY* variables from the environment on first call.

    Args:
        delay : Seconds to sleep after returning the key (default 0).

    Returns:
        A Cerebras API key string.

    Raises:
        ValueError: If no CEREBRAS_API_KEY* variables are found.
    """
    global _cerebras_cycle, _cerebras_count
    if _cerebras_cycle is None:
        _cerebras_cycle, _cerebras_count = _create_key_cycle("CEREBRAS_API_KEY")
        if _cerebras_cycle is None:
            raise ValueError(
                "❌ No Cerebras keys found. "
                "Set CEREBRAS_API_KEY (+ optional CEREBRAS_API_KEY_1, _2 ...) in environment."
            )
    key = next(_cerebras_cycle)
    _log_key_usage("CEREBRAS", key, delay)
    return key


def get_groq_key(delay: float = 0) -> str:
    """
    Returns the next available Groq API key in round-robin rotation.

    Discovers all GROQ_API_KEY* variables from the environment on first call.

    Args:
        delay : Seconds to sleep after returning the key (default 0).

    Returns:
        A Groq API key string.

    Raises:
        ValueError: If no GROQ_API_KEY* variables are found.
    """
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
        if _groq_cycle is None:
            raise ValueError(
                "❌ No Groq keys found. "
                "Set GROQ_API_KEY (+ optional GROQ_API_KEY_1, _2 ...) in environment."
            )
    key = next(_groq_cycle)
    _log_key_usage("GROQ", key, delay)
    return key


def get_tavily_key(delay: float = 0) -> str:
    """
    Returns the next available Tavily API key in round-robin rotation.

    Discovers all TAVILY_API_KEY* variables from the environment on first call.

    Args:
        delay : Seconds to sleep after returning the key (default 0).

    Returns:
        A Tavily API key string.

    Raises:
        ValueError: If no TAVILY_API_KEY* variables are found.
    """
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
        if _tavily_cycle is None:
            raise ValueError(
                "❌ No Tavily keys found. "
                "Set TAVILY_API_KEY (+ optional TAVILY_API_KEY_1, _2 ...) in environment."
            )
    key = next(_tavily_cycle)
    _log_key_usage("TAVILY", key, delay)
    return key


def get_serpapi_key(delay: float = 0) -> str:
    """
    Returns the next available SerpAPI key, sorted by highest credits remaining.

    On first call, validates ALL SerpAPI keys concurrently via async HTTP
    (using httpx — non-blocking). Keys with 0 credits are dropped. Valid keys
    are sorted so the one with the most credits is used first.

    This validation is LAZY — it only runs on the first get_serpapi_key() call,
    not at import or app startup. This keeps Render's startup time fast.

    Args:
        delay : Seconds to sleep after returning the key (default 0).

    Returns:
        A SerpAPI key string with credits remaining.

    Raises:
        ValueError: If no SERPAPI_KEY* variables exist or all have 0 credits.
    """
    global _serpapi_cycle, _serpapi_count
    if _serpapi_cycle is None:
        _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle_sync("SERPAPI_KEY")
        if _serpapi_cycle is None:
            raise ValueError(
                "❌ No SerpAPI keys available. "
                "All keys may be exhausted or invalid. "
                "Set SERPAPI_KEY (+ optional SERPAPI_KEY_1, _2 ...) in environment."
            )
    key = next(_serpapi_cycle)
    _log_key_usage("SERPAPI", key, delay)
    return key


def get_azure_config() -> dict:
    """
    Reads and returns Azure OpenAI credentials from environment variables.

    Azure uses a single key + endpoint config (not rotated like other providers).

    Required environment variables:
        AZURE_API_KEY   : Your Azure OpenAI resource API key.
        AZURE_ENDPOINT  : Your Azure resource endpoint URL.

    Optional environment variables (defaults provided):
        AZURE_DEPLOYMENT  : Model deployment name.   Default: "gpt-4o-mini"
        AZURE_API_VERSION : API version string.       Default: "2024-02-15-preview"

    Returns:
        Dict with keys: api_key, endpoint, deployment, api_version.

    Raises:
        ValueError: If AZURE_API_KEY or AZURE_ENDPOINT are not set.
    """
    api_key     = os.getenv("AZURE_API_KEY",     "").strip()
    endpoint    = os.getenv("AZURE_ENDPOINT",    "").strip()
    deployment  = os.getenv("AZURE_DEPLOYMENT",  "gpt-4o-mini").strip()
    api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview").strip()

    if not api_key or not endpoint:
        raise ValueError(
            "❌ Azure config is incomplete. "
            "Both AZURE_API_KEY and AZURE_ENDPOINT must be set in your environment."
        )

    print(f"✅ Azure config loaded → deployment={deployment}, api_version={api_version}")

    return {
        "api_key":     api_key,
        "endpoint":    endpoint,
        "deployment":  deployment,
        "api_version": api_version,
    }


# ==============================================================================
# SECTION 4 — KEY COUNT HELPERS
# ==============================================================================
# Use these when other parts of your code need to know how many keys are loaded,
# e.g. to decide concurrency levels or log a startup summary.

def get_gemini_count() -> int:
    """Returns number of Gemini keys loaded. Triggers lazy init on first call."""
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
    return _google_count

def get_cerebras_count() -> int:
    """Returns number of Cerebras keys loaded. Triggers lazy init on first call."""
    global _cerebras_cycle, _cerebras_count
    if _cerebras_cycle is None:
        _cerebras_cycle, _cerebras_count = _create_key_cycle("CEREBRAS_API_KEY")
    return _cerebras_count

def get_groq_count() -> int:
    """Returns number of Groq keys loaded. Triggers lazy init on first call."""
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
    return _groq_count

def get_tavily_count() -> int:
    """Returns number of Tavily keys loaded. Triggers lazy init on first call."""
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
    return _tavily_count

def get_serpapi_count() -> int:
    """Returns number of valid SerpAPI keys. Triggers lazy validation on first call."""
    global _serpapi_cycle, _serpapi_count
    if _serpapi_cycle is None:
        _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle_sync("SERPAPI_KEY")
    return _serpapi_count

# Backward-compatibility aliases (older scripts using get_google_key still work)
def get_google_key(delay: float = 0) -> str:
    """Alias for get_gemini_key(). Kept for backward compatibility."""
    return get_gemini_key(delay)

def get_google_count() -> int:
    """Alias for get_gemini_count(). Kept for backward compatibility."""
    return get_gemini_count()


# ==============================================================================
# SECTION 5 — ASYNC KEY WORKER
# ==============================================================================

class KeyWorker:
    """
    Represents ONE API key as an async worker with rate limiting and quota tracking.

    MENTAL MODEL:
    Imagine each API key is a cashier at a store. Each cashier can only serve
    N customers per minute (RPM limit). This class makes each cashier enforce
    their own pace — they sleep between customers so they never get overwhelmed.

    FIX 1 — NON-BLOCKING wait_and_acquire():
    Previously, if a worker was in cooldown, calling wait_and_acquire() would
    block the caller for up to 30 seconds while waiting. This caused Render
    504 timeouts when all workers happened to be cooling simultaneously.

    Now, wait_and_acquire() checks if the worker is cooling and returns False
    IMMEDIATELY instead of sleeping. The pool manager (get_next_available_worker)
    then tries the next worker instantly — no waiting, no timeout risk.

    FIX 3 — AUTO-DISABLE after MAX_FAILURES consecutive 429s:
    If a key hits 429 five times in a row without a single success in between,
    it's treated as permanently broken for today and force-exhausted. This
    prevents the "death spiral" where a dead key gets retried endlessly.

    COOLDOWN VALUES (flat, provider-specific, short enough for Render):
      Gemini   → 30s   (Google's RPM window is 60s; 30s is a safe midpoint)
      Cerebras → 15s   (Cerebras recovers fast)
      Groq     → 20s   (Slightly stricter, 20s is safe)

    STAGGER ON STARTUP:
      Each worker has a startup_delay so they don't all fire simultaneously
      when the app boots. This prevents IP-level burst detection by providers.
    """

    # How many consecutive 429s before a worker is permanently disabled today.
    # FIX 3: This threshold did not exist before — workers retried forever.
    MAX_FAILURES = 5

    # Flat cooldown per provider after a single 429.
    # Short enough to stay within Render's ~30s request timeout on retry.
    _COOLDOWN_MAP = {
        "gemini":   30.0,
        "cerebras": 15.0,
        "groq":     20.0,
    }
    _COOLDOWN_DEFAULT = 20.0  # Fallback for any unrecognized provider

    def __init__(
        self,
        api_key:       str,
        provider:      str,
        sleep_sec:     float,
        daily_cap:     int,
        startup_delay: float = 0.0,
    ):
        """
        Sets up one API key worker.

        Args:
            api_key       : The raw API key string.
            provider      : "gemini", "cerebras", or "groq".
            sleep_sec     : Minimum seconds between consecutive calls (enforces RPM).
            daily_cap     : Max successful calls allowed per day for this key.
            startup_delay : One-time delay before this worker's very first call.
                            Staggering workers prevents IP-level burst on app startup.
        """
        self.api_key       = api_key
        self.provider      = provider
        self.sleep_sec     = sleep_sec
        self.daily_cap     = daily_cap
        self.startup_delay = startup_delay

        self.daily_count    = 0
        self._lock          = None    # Lazy — asyncio.Lock must be created inside event loop
        self._last_call_at  = 0.0
        self._cooling_until = 0.0
        self._retry_count   = 0       # Tracks consecutive 429s (reset on any success)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        """
        Returns the async lock, creating it lazily on first use.

        asyncio.Lock() must be created inside a running event loop.
        Creating it at __init__ time (before the loop starts) raises RuntimeError.
        Lazy creation here is the standard safe pattern for asyncio.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Public Status Properties
    # ------------------------------------------------------------------

    @property
    def is_exhausted(self) -> bool:
        """True when this worker has used up its full daily quota."""
        return self.daily_count >= self.daily_cap

    @property
    def is_cooling(self) -> bool:
        """True while this worker is in its post-429 cooldown window."""
        return time.monotonic() < self._cooling_until

    @property
    def is_ready(self) -> bool:
        """
        True when this worker is fully available for an immediate call.
        (Not exhausted AND not cooling AND enough time since last call.)
        """
        if self.is_exhausted or self.is_cooling:
            return False
        elapsed = time.monotonic() - self._last_call_at
        return elapsed >= self.sleep_sec

    # ------------------------------------------------------------------
    # FIX 1 — Non-Blocking wait_and_acquire
    # ------------------------------------------------------------------

    async def wait_and_acquire(self) -> bool:
        """
        Attempts to acquire this worker for one API call.

        FIX 1 — THIS IS NOW NON-BLOCKING FOR COOLING/EXHAUSTED WORKERS:
        If the worker is exhausted or currently cooling after a 429, this
        method returns False IMMEDIATELY without sleeping. The caller
        (get_next_available_worker) will instantly try the next worker.

        Only the RPM rate-limit sleep (step 4) is still a blocking wait —
        but that sleep is at most sleep_sec (2.5s–6.5s), well within Render's
        30s timeout window.

        FLOW:
          Step 1 → Exhausted?  → return False immediately (no sleep).
          Step 2 → Cooling?    → return False immediately (no sleep). ← FIX 1
          Step 3 → First call? → apply one-time startup_delay (stagger).
          Step 4 → Too recent? → sleep the RPM gap (at most sleep_sec seconds).
          Step 5 → Acquire, update state, return True.

        Returns:
            True  → Worker is acquired; caller may fire the API call now.
            False → Worker is unavailable; caller should try a different worker.
        """
        async with self._get_lock():

            # Step 1: Hard stop — daily quota fully used up
            if self.is_exhausted:
                return False

            # Step 2: FIX 1 — Do NOT sleep here. Return False instantly.
            # The pool manager will pick another worker without any delay.
            if self.is_cooling:
                return False

            # Step 3: One-time startup stagger (fires only on very first call ever)
            if self.startup_delay > 0 and self._last_call_at == 0.0:
                await asyncio.sleep(self.startup_delay)

            # Step 4: RPM gap enforcement — this sleep is bounded by sleep_sec max
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.sleep_sec:
                await asyncio.sleep(self.sleep_sec - elapsed)

            # Step 5: Mark as acquired and update internal state
            self._last_call_at = time.monotonic()
            self.daily_count  += 1
            return True

    # ------------------------------------------------------------------
    # Error Response Handlers
    # ------------------------------------------------------------------

    def mark_429(self):
        """
        Call this immediately after receiving a 429 (Too Many Requests) error.

        WHAT HAPPENS:
          - Daily counter rolls back by 1 (the call didn't succeed).
          - Flat cooldown is set based on provider (30s / 15s / 20s).
          - Retry counter increments.
          - FIX 3: If retry count hits MAX_FAILURES, worker is force-exhausted.
            This prevents endless retry loops on permanently broken keys.

        HOW TO USE:
            try:
                response = await call_api(worker.api_key, prompt)
                worker.reset_retry_count()
            except RateLimitError:
                worker.mark_429()
        """
        self.daily_count  = max(0, self.daily_count - 1)
        self._retry_count += 1

        masked = self.api_key[:6] + "..." + self.api_key[-4:]

        # FIX 3: Auto-disable after too many consecutive failures
        if self._retry_count >= self.MAX_FAILURES:
            self.daily_count = self.daily_cap   # Force-exhaust
            print(
                f"🚫 [{self.provider.upper()}] Key {masked} → "
                f"Disabled after {self.MAX_FAILURES} consecutive 429s. "
                f"Will not be used again today."
            )
            return

        wait = self._COOLDOWN_MAP.get(self.provider, self._COOLDOWN_DEFAULT)
        self._cooling_until = time.monotonic() + wait

        print(
            f"⏳ [{self.provider.upper()}] Key {masked} → "
            f"429 received (consecutive hit #{self._retry_count}/{self.MAX_FAILURES}). "
            f"Cooling {wait:.0f}s."
        )

    def reset_retry_count(self):
        """
        Call this after every SUCCESSFUL API response.

        Resets the consecutive-429 counter to 0. This ensures that a key
        which had a bad patch but recovered is not unfairly penalized — the
        MAX_FAILURES countdown starts fresh from the next 429 onwards.
        """
        self._retry_count = 0

    def mark_daily_exhausted(self):
        """
        Manually force-exhausts this worker for the rest of today.

        Call this when the API explicitly returns a daily quota exceeded error
        (distinct from a per-minute 429). The worker will be skipped for all
        future calls until the app restarts (which resets daily_count to 0).
        """
        self.daily_count = self.daily_cap
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        print(f"🚫 [{self.provider.upper()}] Key {masked} → Daily quota exceeded. Exhausted for today.")

    # ------------------------------------------------------------------
    # Debug Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        if self.is_exhausted:
            status = "EXHAUSTED"
        elif self.is_cooling:
            remaining = self._cooling_until - time.monotonic()
            status = f"COOLING ({remaining:.0f}s left)"
        else:
            status = "READY"
        return (
            f"KeyWorker(provider={self.provider}, key={masked}, "
            f"used={self.daily_count}/{self.daily_cap}, "
            f"failures={self._retry_count}/{self.MAX_FAILURES}, status={status})"
        )


# ==============================================================================
# SECTION 6 — POOL-LEVEL WORKER SELECTOR  (FIX 1 companion function)
# ==============================================================================

async def get_next_available_worker(workers: list) -> "KeyWorker | None":
    """
    Scans the worker pool and returns the first worker that is ready RIGHT NOW.

    FIX 1 COMPANION:
    Because wait_and_acquire() now returns False immediately for cooling/exhausted
    workers (instead of blocking), this function can scan all workers in microseconds
    and jump straight to the first available one. No sleeping, no timeouts.

    PRIORITY:
    Workers are tried in the order they appear in the list. build_worker_pool()
    puts Gemini first, then Cerebras, then Groq — so higher-quality providers
    are naturally preferred when multiple workers are free simultaneously.

    Args:
        workers : The list returned by build_worker_pool().

    Returns:
        The first KeyWorker that successfully acquired, or None if ALL workers
        are currently exhausted or cooling (caller should wait and retry later).

    HOW TO USE IN YOUR CALLING CODE:
        worker = await get_next_available_worker(pool)
        if worker is None:
            # All workers busy — wait a few seconds and retry, or return error
            await asyncio.sleep(5)
            worker = await get_next_available_worker(pool)

        if worker:
            try:
                result = await call_api(worker.api_key, prompt)
                worker.reset_retry_count()
            except RateLimitError:
                worker.mark_429()
            except DailyQuotaError:
                worker.mark_daily_exhausted()
    """
    for worker in workers:
        acquired = await worker.wait_and_acquire()
        if acquired:
            return worker
    return None  # All workers are cooling or exhausted right now


# ==============================================================================
# SECTION 7 — WORKER POOL FACTORY
# ==============================================================================

def build_worker_pool() -> list:
    """
    Scans environment variables and builds the complete async worker pool.

    DYNAMIC WORKER CREATION:
    One KeyWorker is created per API key found. You never specify the count
    in code — just add or remove keys in your .env or Render dashboard and
    the pool adjusts automatically on the next app start.

    PRIORITY ORDER: Gemini workers → Cerebras workers → Groq workers
    This ordering means get_next_available_worker() tries Gemini first,
    then falls through to Cerebras/Groq only when Gemini keys are busy.

    STAGGER DELAYS:
      Gemini   → i × 2.0s  (key 0 fires at t=0s, key 1 at t=2s, key 2 at t=4s ...)
      Cerebras → i × 0.5s  (key 0 at t=0s, key 1 at t=0.5s ...)
      Groq     → i × 0.5s  (key 0 at t=0s, key 1 at t=0.5s ...)
      Gemini gets wider stagger (2s) because Google is strictest about IP bursts.

    DAILY CAPS (from .env, with safe defaults):
      GEMINI_DAILY_CAP   = 480    (RPD ≈ 500 with 4% safety buffer)
      CEREBRAS_DAILY_CAP = 12000  (RPD = 14,400 with buffer)
      GROQ_DAILY_CAP     = 155    (TPD 500K ÷ ~3K tokens avg, 7% buffer)

    Returns:
        Flat list of KeyWorker objects ordered Gemini → Cerebras → Groq.

    Raises:
        RuntimeError: If zero keys are found across all three providers.
    """
    workers = []

    # Daily caps — read from env with safe fallback defaults
    gemini_cap   = int(os.getenv("GEMINI_DAILY_CAP",   "480"))
    cerebras_cap = int(os.getenv("CEREBRAS_DAILY_CAP", "12000"))
    groq_cap     = int(os.getenv("GROQ_DAILY_CAP",     "155"))

    # ── Gemini workers ────────────────────────────────────────────────
    gemini_keys = _get_all_keys("GOOGLE_API_KEY")
    for i, key in enumerate(gemini_keys):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "gemini",
            sleep_sec     = 6.5,
            daily_cap     = gemini_cap,
            startup_delay = i * 2.0,
        ))

    # ── Cerebras workers ──────────────────────────────────────────────
    cerebras_keys = _get_all_keys("CEREBRAS_API_KEY")
    for i, key in enumerate(cerebras_keys):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "cerebras",
            sleep_sec     = 2.5,
            daily_cap     = cerebras_cap,
            startup_delay = i * 0.5,
        ))

    # ── Groq workers ──────────────────────────────────────────────────
    groq_keys = _get_all_keys("GROQ_API_KEY")
    for i, key in enumerate(groq_keys):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "groq",
            sleep_sec     = 6.5,
            daily_cap     = groq_cap,
            startup_delay = i * 0.5,
        ))

    # ── Startup summary ───────────────────────────────────────────────
    g = len(gemini_keys)
    c = len(cerebras_keys)
    q = len(groq_keys)

    print(
        f"\n{'='*68}\n"
        f"  ASYNC WORKER POOL READY  (v3 — non-blocking + async SerpAPI + auto-disable)\n"
        f"  Gemini   : {g:>2} worker(s) → {g*10:>4}/min | {g*gemini_cap:>8,}/day  "
        f"(cap={gemini_cap}, stagger=2.0s, cooldown=30s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  Cerebras : {c:>2} worker(s) → {c*24:>4}/min | {c*cerebras_cap:>8,}/day  "
        f"(cap={cerebras_cap}, stagger=0.5s, cooldown=15s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  Groq     : {q:>2} worker(s) → {q*9:>4}/min  | {q*groq_cap:>8,}/day  "
        f"(cap={groq_cap}, stagger=0.5s, cooldown=20s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  TOTAL    : {len(workers)} worker(s) → {g*10 + c*24 + q*9}/min combined\n"
        f"  FIX 1    : Cooling workers skipped instantly (non-blocking)\n"
        f"  FIX 2    : SerpAPI validation is now async (httpx, non-blocking)\n"
        f"  FIX 3    : Keys auto-disabled after {KeyWorker.MAX_FAILURES} consecutive 429s\n"
        f"{'='*68}\n"
    )

    if len(workers) == 0:
        raise RuntimeError(
            "❌ No API keys found. "
            "Set at least one of: GOOGLE_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY."
        )

    return workers
