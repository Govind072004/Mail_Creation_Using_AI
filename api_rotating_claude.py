
import os
import asyncio
import itertools
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# SECTION 1 — LEGACY SYNC HELPERS  (kept 100% intact for other scripts)
# ==============================================================================

def _log_key_usage(service_name: str, key: str, delay: float = 0):
    masked_key = key[:6] + "..." + key[-4:]
    timestamp  = datetime.now().strftime("%H:%M:%S")
    print(f"🔁 [{timestamp}] {service_name} → Using API Key: {masked_key}")
    if delay > 0:
        print(f"⏳ {service_name} → Waiting {delay}s before next rotation...")
        time.sleep(delay)


def _create_key_cycle(prefix: str):
    keys = []
    for env_key, env_value in os.environ.items():
        if (env_key == prefix or env_key.startswith(f"{prefix}_")) and env_value.strip():
            keys.append(env_value.strip())
    if not keys:
        print(f"⚠️  Warning: No API keys found for {prefix}")
        return None, 0
    print(f"✅ Key Manager: Loaded {len(keys)} keys for {prefix}")
    return itertools.cycle(keys), len(keys)


# ── Azure OpenAI config reader ─────────────────────────────────
def get_azure_config() -> dict:
    """
    Reads Azure credentials from .env
    Matches exactly the test script values.
    """
    api_key     = os.getenv("AZURE_API_KEY",     "").strip()
    endpoint    = os.getenv("AZURE_ENDPOINT",    "").strip()
    deployment  = os.getenv("AZURE_DEPLOYMENT",  "gpt-4o-mini").strip()
    api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview").strip()

    if not api_key or not endpoint:
        raise ValueError(
            "❌ Azure config missing in .env!\n"
            "   AZURE_API_KEY aur AZURE_ENDPOINT dono required hain."
        )

    print(
        f"✅ Azure Config Loaded → "
        f"deployment={deployment}, "
        f"api_version={api_version}"
    )

    return {
        "api_key":     api_key,
        "endpoint":    endpoint,
        "deployment":  deployment,
        "api_version": api_version,
    }
def _create_smart_serpapi_cycle(prefix: str):
    raw_keys = []
    for env_key, env_value in os.environ.items():
        if (env_key == prefix or env_key.startswith(f"{prefix}_")) and env_value.strip():
            raw_keys.append(env_value.strip())
    if not raw_keys:
        print(f"⚠️  Warning: No API keys found for {prefix}")
        return None, 0

    print(f"🔄 Validating {len(raw_keys)} {prefix} keys…")
    valid_keys_info = []
    for key in raw_keys:
        try:
            response = requests.get(f"https://serpapi.com/account?api_key={key}", timeout=5)
            data = response.json()
            if "error" not in data and response.status_code == 200:
                searches_left = data.get("total_searches_left", 0)
                if searches_left > 0:
                    valid_keys_info.append({"key": key, "credits": searches_left})
        except Exception:
            continue

    if not valid_keys_info:
        print(f"❌ Critical: All {prefix} keys are invalid or have 0 credits.")
        return None, 0

    valid_keys_info.sort(key=lambda x: x["credits"], reverse=True)
    sorted_keys = [item["key"] for item in valid_keys_info]
    print(f"✅ Loaded {len(sorted_keys)} ACTIVE {prefix} keys (sorted by highest credits).")
    print(f"🏆 Top key has {valid_keys_info[0]['credits']} credits left.")
    return itertools.cycle(sorted_keys), len(sorted_keys)


# ------------------------------------------------------------------
# LAZY INIT GLOBALS (legacy)
# ------------------------------------------------------------------
_groq_cycle,    _groq_count    = None, 0
_tavily_cycle,  _tavily_count  = None, 0
_google_cycle,  _google_count  = None, 0
_serpapi_cycle, _serpapi_count = None, 0


def get_groq_key(delay: float = 0):
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
        if _groq_cycle is None:
            raise ValueError("No GROQ API keys configured")
    key = next(_groq_cycle)
    _log_key_usage("GROQ", key, delay)
    return key


def get_tavily_key(delay: float = 0):
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
        if _tavily_cycle is None:
            raise ValueError("No TAVILY API keys configured")
    key = next(_tavily_cycle)
    _log_key_usage("TAVILY", key, delay)
    return key


def get_google_key(delay: float = 0):
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
        if _google_cycle is None:
            raise ValueError("No GOOGLE API keys configured")
    key = next(_google_cycle)
    _log_key_usage("GOOGLE", key, delay)
    return key


def get_serpapi_key(delay: float = 0):
    global _serpapi_cycle, _serpapi_count
    if _serpapi_cycle is None:
        _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle("SERPAPI_KEY")
        if _serpapi_cycle is None:
            raise ValueError("No SERPAPI keys configured. They might all be exhausted.")
    key = next(_serpapi_cycle)
    _log_key_usage("SERPAPI", key, delay)
    return key


def get_groq_count():
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
    return _groq_count

def get_tavily_count():
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
    return _tavily_count

def get_serpapi_count():
    global _serpapi_cycle, _serpapi_count
    if _serpapi_cycle is None:
        _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle("SERPAPI_KEY")
    return _serpapi_count

def get_google_count():
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
    return _google_count


# ==============================================================================
# SECTION 2 — NEW ASYNC WORKER POOL  (for 3-API unified queue system)
# ==============================================================================

class KeyWorker:
    """
    Represents ONE API key with built-in async rate limiting + daily quota tracking.

    Each KeyWorker:
      - Owns exactly one API key.
      - Enforces its own RPM sleep so it can NEVER trigger a 429 proactively.
      - Tracks daily usage and stops itself when the daily cap is reached.
      - Handles 429 responses with EXPONENTIAL BACKOFF (not flat cooldown).
      - Staggers its first call using startup_delay to avoid IP burst detection.

    Verified Rate Limits (official docs, March 2026):
    ┌──────────────────────┬────────┬──────────┬──────────────┬────────────┐
    │ Provider / Model     │  RPM   │   TPM    │  Daily Cap   │  Sleep (s) │
    ├──────────────────────┼────────┼──────────┼──────────────┼────────────┤
    │ Gemini 2.5 Flash     │  10    │ Generous │  .env value  │   6.5      │
    │ Cerebras gpt-oss-120b│  30    │  64 K    │  .env value  │   3.0      │
    │ Groq llama-4-scout   │  30    │  30 K    │  .env value  │   6.5      │
    └──────────────────────┴────────┴──────────┴──────────────┴────────────┘

    CHANGE 1 — Staggered Startup:
      startup_delay staggers first API hit per worker so all keys don't
      fire simultaneously (avoids IP-level burst detection by Google).
      Gemini:   0s, 1s, 2s … 8s  (1s gap between each key)
      Cerebras: 0s, 1.5s, 3s     (1.5s gap)
      Groq:     0s, 1s, 2s … 5s  (1s gap)

    CHANGE 2 — Dynamic Daily Cap:
      daily_cap is now read from .env at runtime, not hardcoded.
      Set in .env: GEMINI_DAILY_CAP, CEREBRAS_DAILY_CAP, GROQ_DAILY_CAP

    CHANGE 3 — Exponential Backoff on 429:
      1st 429 →  65s wait
      2nd 429 → 130s wait
      3rd 429 → 260s wait
      4th 429 → 600s wait  (max cap, never more)
      On success → retry counter resets to 0
    """

    # ── CHANGE 3: Exponential backoff constants ──────────────────────
    _BACKOFF_BASE    = 65.0   # seconds for 1st 429
    _BACKOFF_MAX     = 600.0  # hard ceiling (10 minutes)

    def __init__(
        self,
        api_key:       str,
        provider:      str,
        sleep_sec:     float,
        daily_cap:     int,
        startup_delay: float = 0.0,   # ← CHANGE 1
    ):
        self.api_key       = api_key
        self.provider      = provider      # "gemini" | "cerebras" | "groq"
        self.sleep_sec     = sleep_sec
        self.daily_cap     = daily_cap
        self.startup_delay = startup_delay  # ← CHANGE 1

        self.daily_count    = 0
        self._lock          = None          # created lazily inside event loop
        self._last_call_at  = 0.0
        self._cooling_until = 0.0
        self._retry_count   = 0             # ← CHANGE 3: tracks consecutive 429s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_exhausted(self) -> bool:
        """True when daily quota is fully consumed."""
        return self.daily_count >= self.daily_cap

    @property
    def is_cooling(self) -> bool:
        """True while serving a 429 cool-down period."""
        return time.monotonic() < self._cooling_until

    # ------------------------------------------------------------------
    # Core acquire method
    # ------------------------------------------------------------------

    async def wait_and_acquire(self) -> bool:
        """
        Blocks until this key is ready for the next call, then marks it acquired.
        Returns:
            True  → caller may proceed with the API call.
            False → daily quota exhausted; caller must use a different worker.
        """
        async with self._get_lock():

            # ① Daily quota guard
            if self.is_exhausted:
                return False

            # ① b — CHANGE 1: Startup stagger (runs only ONCE on very first call)
            if self.startup_delay > 0 and self._last_call_at == 0.0:
                await asyncio.sleep(self.startup_delay)

            # ② Cool-down wait (after a 429 — uses exponential backoff value)
            now = time.monotonic()
            if now < self._cooling_until:
                await asyncio.sleep(self._cooling_until - now)

            # ③ Rate-limit sleep (enforces RPM / TPM budget)
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.sleep_sec:
                await asyncio.sleep(self.sleep_sec - elapsed)

            # ④ Mark as acquired
            self._last_call_at = time.monotonic()
            self.daily_count  += 1
            return True

    # ------------------------------------------------------------------
    # Error-response handlers
    # ------------------------------------------------------------------

    def mark_429(self):
        """
        CHANGE 3 — Exponential Backoff.
        Call immediately after receiving a 429 response.

        Backoff schedule:
          1st 429 →  65s
          2nd 429 → 130s
          3rd 429 → 260s
          4th 429 → 600s  (capped)

        Also rolls back the daily counter since the call didn't succeed.
        """
        self.daily_count  = max(0, self.daily_count - 1)
        self._retry_count += 1

        wait = min(
            self._BACKOFF_BASE * (2 ** (self._retry_count - 1)),
            self._BACKOFF_MAX,
        )
        self._cooling_until = time.monotonic() + wait

        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        print(
            f"⏳ [{self.provider.upper()}] Key {masked} → "
            f"429 hit #{self._retry_count}. Cooling {wait:.0f}s "
            f"(exponential backoff)."
        )

    def reset_retry_count(self):
        """
        CHANGE 3 — Call this on every successful API response.
        Resets the exponential backoff counter so next 429 starts fresh at 65s.
        """
        self._retry_count = 0

    def mark_daily_exhausted(self):
        """Force-exhausts this key (e.g., when the API returns a daily-limit error)."""
        self.daily_count = self.daily_cap

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        status = (
            "EXHAUSTED" if self.is_exhausted
            else ("COOLING"   if self.is_cooling
            else  "READY")
        )
        return (
            f"KeyWorker(provider={self.provider}, key={masked}, "
            f"used={self.daily_count}/{self.daily_cap}, "
            f"retries={self._retry_count}, status={status})"
        )


# ------------------------------------------------------------------
# Helper — collect all keys for a given env-var prefix
# ------------------------------------------------------------------

def _get_all_keys(prefix: str) -> list:
    """
    Scans os.environ for PREFIX, PREFIX_1, PREFIX_2, …
    Returns a sorted list of non-empty string values.
    """
    keys = []
    for env_name, env_val in os.environ.items():
        if (env_name == prefix or env_name.startswith(f"{prefix}_")) and env_val.strip():
            keys.append(env_val.strip())
    return keys


# ------------------------------------------------------------------
# Main factory — build the full worker pool
# ------------------------------------------------------------------

def build_worker_pool() -> list:
    """
    Reads all API keys + daily caps from .env and returns KeyWorker list.

    Priority order = Gemini → Cerebras → Groq.
    Workers are listed in this order so the queue naturally prefers
    higher-quality APIs first when all are free simultaneously.

    CHANGE 2 — Daily caps read from .env (with safe fallback defaults):
        GEMINI_DAILY_CAP   = 480     (RPD≈500, -4% safety buffer)
        CEREBRAS_DAILY_CAP = 12000   (RPD=14,400 per key)
        GROQ_DAILY_CAP     = 155     (TPD 500K ÷ 3K tokens - 7% buffer)

    CHANGE 1 — Stagger delays per provider:
        Gemini   →  i × 1.0s   (key 0=0s, key 1=1s … key 8=8s)
        Cerebras →  i × 1.5s   (key 0=0s, key 1=1.5s, key 2=3s)
        Groq     →  i × 1.0s   (key 0=0s, key 1=1s … key 5=5s)

    Expected .env variables:
        GOOGLE_API_KEY, GOOGLE_API_KEY_1 … GOOGLE_API_KEY_8   (9 Gemini keys)
        CEREBRAS_API_KEY, CEREBRAS_API_KEY_1, CEREBRAS_API_KEY_2 (3 keys)
        GROQ_API_KEY, GROQ_API_KEY_1 … GROQ_API_KEY_5          (6 keys)

        GEMINI_DAILY_CAP=480
        CEREBRAS_DAILY_CAP=12000
        GROQ_DAILY_CAP=155
    """
    workers = []

    # ── CHANGE 2: Read daily caps from .env (with defaults) ─────────
    gemini_cap   = int(os.getenv("GEMINI_DAILY_CAP",   "480"))
    cerebras_cap = int(os.getenv("CEREBRAS_DAILY_CAP", "12000"))
    groq_cap     = int(os.getenv("GROQ_DAILY_CAP",     "155"))

    # ── Gemini workers ───────────────────────────────────────────────
    # 10 RPM → sleep 6.5s | CHANGE 1: stagger 1s per key
    for i, key in enumerate(_get_all_keys("GOOGLE_API_KEY")):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "gemini",
            sleep_sec     = 6.5,
            daily_cap     = gemini_cap,
            startup_delay = i * 1.0,     # ← CHANGE 1: 0s, 1s, 2s … 8s
        ))

    # ── Cerebras workers ─────────────────────────────────────────────
    # 30 RPM, 64K TPM → sleep 3.0s | CHANGE 1: stagger 1.5s per key
    for i, key in enumerate(_get_all_keys("CEREBRAS_API_KEY")):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "cerebras",
            sleep_sec     = 3.0,
            daily_cap     = cerebras_cap,
            startup_delay = i * 1.5,     # ← CHANGE 1: 0s, 1.5s, 3s
        ))

    # ── Groq workers ─────────────────────────────────────────────────
    # llama-4-scout: 30 RPM, 30K TPM → sleep 6.5s | CHANGE 1: stagger 1s per key
    for i, key in enumerate(_get_all_keys("GROQ_API_KEY")):
        workers.append(KeyWorker(
            api_key       = key,
            provider      = "groq",
            sleep_sec     = 6.5,
            daily_cap     = groq_cap,
            startup_delay = i * 1.0,     # ← CHANGE 1: 0s, 1s, 2s … 5s
        ))

    # ── Summary log ──────────────────────────────────────────────────
    g = sum(1 for w in workers if w.provider == "gemini")
    c = sum(1 for w in workers if w.provider == "cerebras")
    q = sum(1 for w in workers if w.provider == "groq")

    print(
        f"\n{'='*60}\n"
        f"  WORKER POOL BUILT  (v2 — stagger + dynamic caps + backoff)\n"
        f"  Gemini   : {g} workers  →  {g*10}/min  | {g*gemini_cap:,}/day  "
        f"(cap={gemini_cap}, stagger=1.0s)\n"
        f"  Cerebras : {c} workers  →  {c*21}/min  | {c*cerebras_cap:,}/day  "
        f"(cap={cerebras_cap}, stagger=1.5s)\n"
        f"  Groq     : {q} workers  →  {q*9}/min   | {q*groq_cap:,}/day  "
        f"(cap={groq_cap}, stagger=1.0s)\n"
        f"  TOTAL    : {len(workers)} workers  →  "
        f"{g*10 + c*21 + q*9}/min combined\n"
        f"  Startup  : keys fire with 1–1.5s gaps (burst protection ON)\n"
        f"  Backoff  : 65s → 130s → 260s → 600s on repeated 429s\n"
        f"{'='*60}\n"
    )

    if len(workers) == 0:
        raise RuntimeError(
            "❌ No API keys found in .env. "
            "Add GOOGLE_API_KEY, CEREBRAS_API_KEY, and/or GROQ_API_KEY."
        )

    return workers



