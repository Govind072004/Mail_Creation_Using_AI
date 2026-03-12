"""
api_rotating_claude.py
======================
Production-ready API Key Rotation Manager
Designed for Render Free Tier deployment.

VERSION: 3.1 — Fixes applied:
  FIX 1 → wait_and_acquire() is now NON-BLOCKING.
  FIX 2 → SerpAPI validation uses synchronous 'requests' to avoid sniffio/httpx async context crashes.
  FIX 3 → Keys auto-disabled after MAX_FAILURES consecutive 429s.
  FIX 4 → Threading lock added to SerpAPI initialization.
"""

import os
import asyncio
import itertools
import time
import threading
import requests
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Lock to prevent multiple threads from validating SerpAPI keys at the exact same time
_serpapi_lock = threading.Lock()

def _log_key_usage(service_name: str, key: str, delay: float = 0):
    masked_key = key[:6] + "..." + key[-4:]
    timestamp  = datetime.now().strftime("%H:%M:%S")
    print(f"🔁 [{timestamp}] {service_name} → Using key: {masked_key}")
    if delay > 0:
        print(f"⏳ {service_name} → Waiting {delay}s before next rotation...")
        time.sleep(delay)


def _get_all_keys(prefix: str) -> list:
    keys = []
    for env_name, env_val in os.environ.items():
        if (env_name == prefix or env_name.startswith(f"{prefix}_")) and env_val.strip():
            keys.append(env_val.strip())
    return keys


def _create_key_cycle(prefix: str):
    keys = _get_all_keys(prefix)
    if not keys:
        print(f"⚠️  Warning: No keys found for prefix '{prefix}'")
        return None, 0
    print(f"✅ Key Manager: Loaded {len(keys)} key(s) for '{prefix}'")
    return itertools.cycle(keys), len(keys)


def _create_smart_serpapi_cycle_sync(prefix: str):
    """
    Synchronous SerpAPI validator using 'requests'.
    This safely bypasses the httpx/sniffio async context issues in ThreadPoolExecutor.
    """
    raw_keys = _get_all_keys(prefix)
    if not raw_keys:
        print(f"⚠️ No SerpAPI keys found for prefix '{prefix}'")
        return None, 0

    print(f"🔄 Validating {len(raw_keys)} SerpAPI key(s)...")
    valid_keys = []

    for key in raw_keys:
        try:
            r = requests.get(
                "https://serpapi.com/account",
                params={"api_key": key},
                timeout=5
            )
            data = r.json()

            if "error" not in data and r.status_code == 200:
                credits = data.get("total_searches_left", 0)

                if credits > 0:
                    valid_keys.append((key, credits))
                else:
                    masked = key[:6] + "..." + key[-4:]
                    print(f"⚠️ SerpAPI key {masked} → 0 credits")
        except Exception as e:
            masked = key[:6] + "..." + key[-4:]
            print(f"⚠️ SerpAPI key {masked} → Validation failed ({e})")

    if not valid_keys:
        print("❌ No valid SerpAPI keys found.")
        return None, 0

    valid_keys.sort(key=lambda x: x[1], reverse=True)
    sorted_keys = [k[0] for k in valid_keys]

    print(f"✅ SerpAPI: {len(sorted_keys)} active key(s) validated")
    return itertools.cycle(sorted_keys), len(sorted_keys)


_groq_cycle,     _groq_count     = None, 0
_tavily_cycle,   _tavily_count   = None, 0
_google_cycle,   _google_count   = None, 0
_serpapi_cycle,  _serpapi_count  = None, 0
_cerebras_cycle, _cerebras_count = None, 0


def get_gemini_key(delay: float = 0) -> str:
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
        if _google_cycle is None:
            raise ValueError("❌ No Gemini keys found.")
    key = next(_google_cycle)
    _log_key_usage("GEMINI", key, delay)
    return key


def get_cerebras_key(delay: float = 0) -> str:
    global _cerebras_cycle, _cerebras_count
    if _cerebras_cycle is None:
        _cerebras_cycle, _cerebras_count = _create_key_cycle("CEREBRAS_API_KEY")
        if _cerebras_cycle is None:
            raise ValueError("❌ No Cerebras keys found.")
    key = next(_cerebras_cycle)
    _log_key_usage("CEREBRAS", key, delay)
    return key


def get_groq_key(delay: float = 0) -> str:
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
        if _groq_cycle is None:
            raise ValueError("❌ No Groq keys found.")
    key = next(_groq_cycle)
    _log_key_usage("GROQ", key, delay)
    return key


def get_tavily_key(delay: float = 0) -> str:
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
        if _tavily_cycle is None:
            raise ValueError("❌ No Tavily keys found.")
    key = next(_tavily_cycle)
    _log_key_usage("TAVILY", key, delay)
    return key


def get_serpapi_key(delay: float = 0) -> str:
    global _serpapi_cycle, _serpapi_count
    with _serpapi_lock:
        if _serpapi_cycle is None:
            _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle_sync("SERPAPI_KEY")
            if _serpapi_cycle is None:
                raise ValueError("❌ No SerpAPI keys available.")
                
    key = next(_serpapi_cycle)
    _log_key_usage("SERPAPI", key, delay)
    return key


def get_azure_config() -> dict:
    api_key     = os.getenv("AZURE_API_KEY",     "").strip()
    endpoint    = os.getenv("AZURE_ENDPOINT",    "").strip()
    deployment  = os.getenv("AZURE_DEPLOYMENT",  "gpt-4o-mini").strip()
    api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview").strip()

    if not api_key or not endpoint:
        raise ValueError("❌ Azure config is incomplete. Both AZURE_API_KEY and AZURE_ENDPOINT must be set.")

    print(f"✅ Azure config loaded → deployment={deployment}, api_version={api_version}")
    return {
        "api_key":     api_key,
        "endpoint":    endpoint,
        "deployment":  deployment,
        "api_version": api_version,
    }


def get_gemini_count() -> int:
    global _google_cycle, _google_count
    if _google_cycle is None:
        _google_cycle, _google_count = _create_key_cycle("GOOGLE_API_KEY")
    return _google_count

def get_cerebras_count() -> int:
    global _cerebras_cycle, _cerebras_count
    if _cerebras_cycle is None:
        _cerebras_cycle, _cerebras_count = _create_key_cycle("CEREBRAS_API_KEY")
    return _cerebras_count

def get_groq_count() -> int:
    global _groq_cycle, _groq_count
    if _groq_cycle is None:
        _groq_cycle, _groq_count = _create_key_cycle("GROQ_API_KEY")
    return _groq_count

def get_tavily_count() -> int:
    global _tavily_cycle, _tavily_count
    if _tavily_cycle is None:
        _tavily_cycle, _tavily_count = _create_key_cycle("TAVILY_API_KEY")
    return _tavily_count

def get_serpapi_count() -> int:
    global _serpapi_cycle, _serpapi_count
    if _serpapi_cycle is None:
        _serpapi_cycle, _serpapi_count = _create_smart_serpapi_cycle_sync("SERPAPI_KEY")
    return _serpapi_count

def get_google_key(delay: float = 0) -> str:
    return get_gemini_key(delay)

def get_google_count() -> int:
    return get_gemini_count()


class KeyWorker:
    MAX_FAILURES = 5
    _COOLDOWN_MAP = {
        "gemini":   30.0,
        "cerebras": 15.0,
        "groq":     20.0,
    }
    _COOLDOWN_DEFAULT = 20.0

    def __init__(self, api_key, provider, sleep_sec, daily_cap, startup_delay=0.0):
        self.api_key       = api_key
        self.provider      = provider
        self.sleep_sec     = sleep_sec
        self.daily_cap     = daily_cap
        self.startup_delay = startup_delay
        self.daily_count    = 0
        self._lock          = None
        self._last_call_at  = 0.0
        self._cooling_until = 0.0
        self._retry_count   = 0

    def _get_lock(self) -> asyncio.Lock:
        # FIX: asyncio.Lock binds to the running loop at creation time.
        # If loop changes between calls (new event loop per pipeline run),
        # a stale cached Lock causes: RuntimeError('Timeout should be used inside a task')
        # Solution: check if cached lock belongs to current running loop; if not, make a new one.
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if self._lock is None or getattr(self._lock, '_loop', None) is not running_loop:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def is_exhausted(self) -> bool:
        return self.daily_count >= self.daily_cap

    @property
    def is_cooling(self) -> bool:
        return time.monotonic() < self._cooling_until

    @property
    def is_ready(self) -> bool:
        if self.is_exhausted or self.is_cooling:
            return False
        elapsed = time.monotonic() - self._last_call_at
        return elapsed >= self.sleep_sec

    async def wait_and_acquire(self) -> bool:
        async with self._get_lock():
            if self.is_exhausted:
                return False
            if self.is_cooling:
                return False
            if self.startup_delay > 0 and self._last_call_at == 0.0:
                await asyncio.sleep(self.startup_delay)
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.sleep_sec:
                await asyncio.sleep(self.sleep_sec - elapsed)
            self._last_call_at = time.monotonic()
            self.daily_count  += 1
            return True

    def mark_429(self):
        self.daily_count  = max(0, self.daily_count - 1)
        self._retry_count += 1
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        if self._retry_count >= self.MAX_FAILURES:
            self.daily_count = self.daily_cap
            print(f"🚫 [{self.provider.upper()}] Key {masked} → Disabled after {self.MAX_FAILURES} consecutive 429s.")
            return
        wait = self._COOLDOWN_MAP.get(self.provider, self._COOLDOWN_DEFAULT)
        self._cooling_until = time.monotonic() + wait
        print(f"⏳ [{self.provider.upper()}] Key {masked} → 429 received (#{self._retry_count}/{self.MAX_FAILURES}). Cooling {wait:.0f}s.")

    def reset_retry_count(self):
        self._retry_count = 0

    def mark_daily_exhausted(self):
        self.daily_count = self.daily_cap
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        print(f"🚫 [{self.provider.upper()}] Key {masked} → Daily quota exceeded.")

    def __repr__(self) -> str:
        masked = self.api_key[:6] + "..." + self.api_key[-4:]
        if self.is_exhausted:
            status = "EXHAUSTED"
        elif self.is_cooling:
            remaining = self._cooling_until - time.monotonic()
            status = f"COOLING ({remaining:.0f}s left)"
        else:
            status = "READY"
        return (f"KeyWorker(provider={self.provider}, key={masked}, "
                f"used={self.daily_count}/{self.daily_cap}, "
                f"failures={self._retry_count}/{self.MAX_FAILURES}, status={status})")


async def get_next_available_worker(workers: list):
    for worker in workers:
        acquired = await worker.wait_and_acquire()
        if acquired:
            return worker
    return None


def build_worker_pool() -> list:
    workers = []
    gemini_cap   = int(os.getenv("GEMINI_DAILY_CAP",   "480"))
    cerebras_cap = int(os.getenv("CEREBRAS_DAILY_CAP", "12000"))
    groq_cap     = int(os.getenv("GROQ_DAILY_CAP",     "155"))

    gemini_keys = _get_all_keys("GOOGLE_API_KEY")
    for i, key in enumerate(gemini_keys):
        workers.append(KeyWorker(api_key=key, provider="gemini", sleep_sec=6.5, daily_cap=gemini_cap, startup_delay=i * 2.0))

    cerebras_keys = _get_all_keys("CEREBRAS_API_KEY")
    for i, key in enumerate(cerebras_keys):
        workers.append(KeyWorker(api_key=key, provider="cerebras", sleep_sec=2.5, daily_cap=cerebras_cap, startup_delay=i * 0.5))

    groq_keys = _get_all_keys("GROQ_API_KEY")
    for i, key in enumerate(groq_keys):
        workers.append(KeyWorker(api_key=key, provider="groq", sleep_sec=6.5, daily_cap=groq_cap, startup_delay=i * 0.5))

    g = len(gemini_keys)
    c = len(cerebras_keys)
    q = len(groq_keys)

    print(
        f"\n{'='*68}\n"
        f"  ASYNC WORKER POOL READY  (v3.1 — non-blocking + sync SerpAPI + auto-disable)\n"
        f"  Gemini   : {g:>2} worker(s) → {g*10:>4}/min | {g*gemini_cap:>8,}/day  (cap={gemini_cap}, stagger=2.0s, cooldown=30s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  Cerebras : {c:>2} worker(s) → {c*24:>4}/min | {c*cerebras_cap:>8,}/day  (cap={cerebras_cap}, stagger=0.5s, cooldown=15s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  Groq     : {q:>2} worker(s) → {q*9:>4}/min  | {q*groq_cap:>8,}/day  (cap={groq_cap}, stagger=0.5s, cooldown=20s, max_fail={KeyWorker.MAX_FAILURES})\n"
        f"  TOTAL    : {len(workers)} worker(s) → {g*10 + c*24 + q*9}/min combined\n"
        f"  FIX 1    : Cooling workers skipped instantly (non-blocking)\n"
        f"  FIX 2    : SerpAPI validation is now sync to prevent sniffio crashes\n"
        f"  FIX 3    : Keys auto-disabled after {KeyWorker.MAX_FAILURES} consecutive 429s\n"
        f"  FIX 4    : Thread event loop explicitly set\n"
        f"{'='*68}\n"
    )

    if len(workers) == 0:
        raise RuntimeError("❌ No API keys found. Set at least one of: GOOGLE_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY.")

    return workers
