"""Gemini API calls with a disk cache, request pacing and backoff on rate-limit errors.

Every response is stored under the hash of (model, prompt, generation config), so a call is
never paid twice in quota and a stopped run resumes where it left off.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 503}

PostFn = Callable[[str, dict, str], dict]


class QuotaExhausted(RuntimeError):
    """Raised when rate-limit errors persist after every retry."""


def cache_key(model: str, prompt: str, generation_config: dict) -> str:
    """Stable hash of everything that determines the response."""
    payload = json.dumps([model, prompt, generation_config], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def http_post(url: str, body: dict, api_key: str) -> dict:
    """POST a JSON body and return the decoded JSON response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def response_text(response: dict) -> str:
    """Text of the first candidate of a generateContent response."""
    parts = response["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


class GeminiClient:
    """Cached, paced client. `post` is injectable so that tests never touch the network."""

    def __init__(
        self,
        model: str,
        api_key: str,
        cache_dir: Path,
        requests_per_minute: float,
        max_retries: int = 5,
        backoff_seconds: float = 20.0,
        post: PostFn = http_post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.min_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.post = post
        self.sleep = sleep
        self.last_call = float("-inf")
        self.calls_made = 0

    def cache_path(self, prompt: str, generation_config: dict) -> Path:
        return self.cache_dir / f"{cache_key(self.model, prompt, generation_config)}.json"

    def is_cached(self, prompt: str, generation_config: dict) -> bool:
        return self.cache_path(prompt, generation_config).exists()

    def generate(self, prompt: str, generation_config: dict) -> dict:
        """Return the cached response, or call the API (paced, with retries) and cache it."""
        path = self.cache_path(prompt, generation_config)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": generation_config}
        response = self._post_with_retries(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "prompt": prompt, "response": response}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record

    def _post_with_retries(self, body: dict) -> dict:
        url = API_URL.format(model=self.model)
        for attempt in range(self.max_retries + 1):
            wait = self.last_call + self.min_interval - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self.last_call = time.monotonic()
            self.calls_made += 1
            try:
                return self.post(url, body, self.api_key)
            except urllib.error.HTTPError as err:
                if err.code not in RETRY_STATUS or attempt == self.max_retries:
                    if err.code == 429:
                        raise QuotaExhausted("rate limit still hit after retries") from err
                    raise
                self.sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable")
