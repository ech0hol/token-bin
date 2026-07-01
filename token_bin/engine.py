"""token-bin core engine: providers, token waster, and waste reporter."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

# ── Data models ──────────────────────────────────────────────────────────────


@dataclass
class CalibrationData:
    """Result of a calibration request for one provider/model."""

    model: str
    sample_chars: int
    sample_prompt_tokens: int
    sample_total_tokens: int
    chars_per_prompt_token: float
    chars_per_total_token: float
    # New fields for precise overhead tracking
    base_overhead_tokens: int  # tokens consumed by system prompt + minimal completion (no user content)


@dataclass
class WasteReport:
    """Final waste report after precision token consumption."""

    model: str
    provider_name: str
    target_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: int  # actual - target
    error_pct: float
    rounds: int
    duration_seconds: float
    estimated_cost_usd: float
    calibration: Optional[CalibrationData] = None

    @property
    def is_perfect(self) -> bool:
        return abs(self.error) <= 2


# ── Providers ────────────────────────────────────────────────────────────────


class BaseProvider(ABC):
    """Abstract LLM API provider."""

    name: str = "base"

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        """Send chat completion. Must return normalized dict with 'usage' key."""
        ...

    def supports_native_tokenizer(self) -> bool:
        return False

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def estimate_chars_for_prompt_tokens(self, target: int) -> int:
        """Estimate chars needed for `target` prompt tokens (no native tokenizer)."""
        return int(target * 4.0)  # rough default: 4 chars / token


class OpenAIProvider(BaseProvider):
    """OpenAI / Azure / any OpenAI-compatible endpoint with tiktoken."""

    name = "openai"

    _MODEL_ENCODING: dict[str, str] = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4.1": "o200k_base",
        "gpt-4.1-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
    }
    _DEFAULT_ENCODING = "cl100k_base"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        import tiktoken

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        enc_name = self._MODEL_ENCODING.get(model, self._DEFAULT_ENCODING)
        self._enc = tiktoken.get_encoding(enc_name)

    # -- native tokenizer ------------------------------------------------------

    def supports_native_tokenizer(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def generate_exact_token_text(self, num_tokens: int) -> str:
        """Generate random text that encodes to *exactly* `num_tokens` tokens."""
        if num_tokens <= 0:
            return ""
        token_ids = [random.randint(0, 50000) for _ in range(num_tokens)]
        text = self._enc.decode(token_ids)

        actual = len(self._enc.encode(text))
        while actual < num_tokens:
            text += self._enc.decode([random.randint(0, 50000)])
            actual = len(self._enc.encode(text))
        while actual > num_tokens:
            tokens = self._enc.encode(text)
            tokens = tokens[:num_tokens]
            text = self._enc.decode(tokens)
            actual = len(self._enc.encode(text))
        return text

    # -- API -------------------------------------------------------------------

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        system = ""
        user_msgs: list[str] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                user_msgs.append(m["content"])

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 256),
            "messages": [{"role": "user", "content": msg} for msg in user_msgs],
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        content_blocks = data.get("content", [{"type": "text", "text": ""}])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": data["usage"]["input_tokens"],
                "completion_tokens": data["usage"]["output_tokens"],
                "total_tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            },
        }


class GenericOpenAIProvider(BaseProvider):
    """Generic OpenAI-compatible API (OpenRouter, Together, local LLMs, etc.)."""

    name = "generic"

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "messages": messages, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()


class DeepSeekProvider(GenericOpenAIProvider):
    """DeepSeek API (OpenAI-compatible). Uses calibration + feedback for precision."""

    name = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_BASE_URL):
        super().__init__(api_key=api_key, model=model, base_url=base_url)


# ── Helpers ──────────────────────────────────────────────────────────────────

_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "by", "from", "they", "we", "say", "her", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them",
    "see", "other", "than", "then", "now", "look", "only", "come", "its",
    "over", "think", "also", "back", "after", "use", "two", "how", "our",
    "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "system", "data", "token",
    "model", "text", "input", "output", "process", "result", "value",
    "each", "many", "part", "place", "point", "world", "life", "hand",
    "number", "group", "problem", "fact", "example", "test", "case",
]

_SYSTEM_PROMPT = (
    "You are a token wastebin. Echo the user message verbatim. "
    "Do not add any commentary, explanation, or extra words. "
    "Output ONLY the exact text the user provided."
)


def _random_text(chars: int) -> str:
    """Generate ~`chars` characters of random word salad."""
    parts: list[str] = []
    total = 0
    while total < chars:
        w = random.choice(_WORDS)
        parts.append(w)
        total += len(w) + 1
    text = " ".join(parts)
    return text[:chars] if len(text) > chars else text


# ── Token Waster Engine ──────────────────────────────────────────────────────

class TokenWaster:
    """Precision token wasting engine with calibration + feedback loop."""

    MAX_ROUNDS = 15
    CONVERGENCE_THRESHOLD = 2  # tokens

    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.calibration: Optional[CalibrationData] = None

    # ── Calibration ──────────────────────────────────────────────────────

    async def calibrate(self) -> CalibrationData:
        """Two-phase calibration: measure overhead + learn chars-per-token ratio.

        Phase 1: Send minimal request (empty-ish user msg) to measure base overhead
                (system prompt tokens + completion tokens).
        Phase 2: Send known-size content to get accurate user-content token ratio.
        """
        # Phase 1: Measure base overhead (system prompt + completion)
        overhead_resp = await self.provider.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "."},  # minimal content
            ],
            max_tokens=1,
            temperature=0,
        )
        base_overhead = overhead_resp["usage"]["total_tokens"]

        # Phase 2: Measure chars -> prompt_tokens ratio for USER content only
        sample_chars = 300
        text = _random_text(sample_chars)

        response = await self.provider.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=1,
            temperature=0,
        )
        usage = response["usage"]
        pt = usage["prompt_tokens"]
        tt = usage["total_tokens"]

        # User-content-only prompt tokens = total_prompt - overhead_prompt
        # We approximate: overhead_prompt ~= base_overhead - 1 (the 1 completion from phase 1)
        est_user_prompt_tokens = max(1, pt - (base_overhead - 1))
        chars_per_user_token = sample_chars / est_user_prompt_tokens

        self.calibration = CalibrationData(
            model=getattr(self.provider, "model", "unknown"),
            sample_chars=sample_chars,
            sample_prompt_tokens=pt,
            sample_total_tokens=tt,
            chars_per_prompt_token=chars_per_user_token,
            chars_per_total_token=sample_chars / tt if tt else 4.0,
            base_overhead_tokens=base_overhead,
        )
        return self.calibration

    # ── Waste ────────────────────────────────────────────────────────────

    async def waste(
        self,
        target_tokens: int,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> WasteReport:
        """Consume `target_tokens` with maximum precision."""
        if target_tokens <= 0:
            return self._empty_report(target_tokens)

        if self.calibration is None:
            await self.calibrate()

        start = time.monotonic()

        if self.provider.supports_native_tokenizer():
            report = await self._waste_native(target_tokens, progress_callback)
        else:
            report = await self._waste_generic(target_tokens, progress_callback)

        report.duration_seconds = time.monotonic() - start
        report.estimated_cost_usd = self._estimate_cost(report)
        return report

    # ── Native tokenizer path (OpenAI + tiktoken) ────────────────────────

    async def _waste_native(
        self, target: int, cb: Optional[Callable[[int, int, int], None]]
    ) -> WasteReport:
        accumulated = 0
        prompt_total = 0
        completion_total = 0
        rounds = 0

        while rounds < self.MAX_ROUNDS:
            remaining = target - accumulated
            if abs(remaining) <= self.CONVERGENCE_THRESHOLD:
                break

            rounds += 1
            prompt_text = self.provider.generate_exact_token_text(
                max(1, remaining - 5)
            )  # -5 for system msg overhead

            resp = await self.provider.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=10,
                temperature=0,
            )
            usage = resp["usage"]
            prompt_total += usage["prompt_tokens"]
            completion_total += usage["completion_tokens"]
            accumulated = prompt_total + completion_total

            if cb:
                cb(rounds, target, accumulated)

        return WasteReport(
            model=getattr(self.provider, "model", "?"),
            provider_name=self.provider.name,
            target_tokens=target,
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
            total_tokens=accumulated,
            error=accumulated - target,
            error_pct=abs(accumulated - target) / target * 100 if target else 0,
            rounds=rounds,
            duration_seconds=0,
            estimated_cost_usd=0,
            calibration=self.calibration,
        )

    # ── Generic path (no native tokenizer) ───────────────────────────────

    async def _waste_generic(
        self, target: int, cb: Optional[Callable[[int, int, int], None]]
    ) -> WasteReport:
        cal = self.calibration  # type: ignore[union-attr]
        cpt = cal.chars_per_prompt_token  # chars per USER CONTENT prompt token
        base_overhead = cal.base_overhead_tokens  # system + completion per call

        accumulated = 0
        prompt_total = 0
        completion_total = 0
        rounds = 0

        # If target is smaller than one call's minimum cost, just do one minimal call
        if target <= base_overhead:
            resp = await self.provider.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": "."},
                ],
                max_tokens=1,
                temperature=0,
            )
            usage = resp["usage"]
            return WasteReport(
                model=getattr(self.provider, "model", "?"),
                provider_name=self.provider.name,
                target_tokens=target,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                error=usage["total_tokens"] - target,
                error_pct=abs(usage["total_tokens"] - target) / target * 100 if target else 0,
                rounds=1,
                duration_seconds=0,
                estimated_cost_usd=0,
                calibration=cal,
            )

        while rounds < self.MAX_ROUNDS:
            remaining = target - accumulated
            if abs(remaining) <= self.CONVERGENCE_THRESHOLD:
                break

            rounds += 1

            # Budget for THIS call: subtract per-call overhead from remaining budget
            usable_for_content = remaining - base_overhead
            if usable_for_content <= 0:
                est_chars = 10
            else:
                est_chars = max(10, int(usable_for_content * cpt))
                est_chars = min(est_chars, 50_000)  # safety cap

            text = _random_text(est_chars)

            resp = await self.provider.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=1,  # minimal completion to reduce variance
                temperature=0,
            )
            usage = resp["usage"]
            prompt_total += usage["prompt_tokens"]
            completion_total += usage["completion_tokens"]
            accumulated = prompt_total + completion_tokens

            # Dynamically refine ratio using user-content-only tokens
            actual_prompt = usage["prompt_tokens"]
            est_user_prompt = max(1, actual_prompt - (base_overhead - 1))
            if est_user_prompt > 0 and len(text) > 0:
                new_cpt = len(text) / est_user_prompt
                cpt = cpt * 0.7 + new_cpt * 0.3  # EMA smooth

            if cb:
                cb(rounds, target, accumulated)

        return WasteReport(
            model=getattr(self.provider, "model", "?"),
            provider_name=self.provider.name,
            target_tokens=target,
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
            total_tokens=accumulated,
            error=accumulated - target,
            error_pct=abs(accumulated - target) / target * 100 if target else 0,
            rounds=rounds,
            duration_seconds=0,
            estimated_cost_usd=0,
            calibration=cal,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _empty_report(self, target: int) -> WasteReport:
        return WasteReport(
            model="?", provider_name="?", target_tokens=0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            error=0, error_pct=0, rounds=0,
            duration_seconds=0, estimated_cost_usd=0,
        )

    @staticmethod
    def _estimate_cost(report: WasteReport) -> float:
        pricing: dict[str, tuple[float, float]] = {
            "gpt-4o": (2.50, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-4": (30.0, 60.0),
            "gpt-3.5-turbo": (0.50, 1.50),
            "claude-3-5-sonnet": (3.0, 15.0),
            "claude-3-5-haiku": (0.80, 4.0),
            "claude-3-opus": (15.0, 75.0),
            "claude-3-haiku": (0.25, 1.25),
            "gemini-2.0-flash": (0.10, 0.40),
            "gemini-1.5-pro": (1.25, 5.0),
            "gemini-1.5-flash": (0.075, 0.30),
            "deepseek-chat": (0.14, 0.28),
            "deepseek-reasoner": (0.55, 2.19),
        }
        m = report.model.lower()
        pp, cp = 1.0, 5.0
        for k, v in pricing.items():
            if k in m:
                pp, cp = v
                break
        return (report.prompt_tokens / 1e6 * pp) + (report.completion_tokens / 1e6 * cp)


# ── Report Formatter ─────────────────────────────────────────────────────────

def format_report(report: WasteReport) -> str:
    """Render a beautiful waste report string (for terminal)."""
    accuracy = 100 - report.error_pct
    status = "PERFECT" if report.is_perfect else "OK"

    lines = [
        "",
        " ╔══════════════════════════════════════════════════╗",
        " ║           TOKEN-BIN WASTE REPORT            ║",
        " ╚══════════════════════════════════════════════════╝",
        "",
        f"  Model          : {report.model}",
        f"  Provider       : {report.provider_name}",
        "",
        "  -- Consumption --",
        f"  Target         : {report.target_tokens:>10,} tokens",
        f"  Actual         : {report.total_tokens:>10,} tokens",
        f"  +- Prompt      : {report.prompt_tokens:>10,} tokens",
        f"  +- Completion  : {report.completion_tokens:>10,} tokens",
        "",
        "  -- Precision --",
        f"  Error          : {report.error:+d} tokens",
        f"  Accuracy       : {accuracy:.2f}%",
        f"  Status         : {status}",
        f"  Rounds         : {report.rounds}",
        f"  Duration       : {report.duration_seconds:.2f}s",
        "",
        "  -- Cost --",
        f"  Estimated      : ${report.estimated_cost_usd:.6f} USD",
    ]

    if report.calibration:
        c = report.calibration
        lines += [
            "",
            "  -- Calibration --",
            f"  Sample chars   : {c.sample_chars}",
            f"  Sample tokens  : {c.sample_prompt_tokens} prompt",
            f"  Chars/token    : {c.chars_per_prompt_token:.2f}",
            f"  Base overhead  : {c.base_overhead_tokens} tokens/call",
        ]

    lines += [
        "",
        " ╔══════════════════════════════════════════════════╗",
        " ║   Token wasted successfully.                      ║",
        " ╚══════════════════════════════════════════════════╝",
        "",
    ]
    return "\n".join(lines)
