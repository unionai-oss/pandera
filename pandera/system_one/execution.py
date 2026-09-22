"""Bounded concurrent fan-out over rows.

One request per (row, question group), dispatched concurrently under a
semaphore and a token bucket sized to the provider's published limits. Results
are restored to input order: a parser must return a frame whose index lines up
with its input, since the answers are joined onto an existing dataframe.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from pandera.system_one.primitives import Decision, ProviderLimits


class TokenBucket:
    """Paces requests against a rate limit.

    Two independent budgets -- requests per minute and tokens per second --
    because providers publish both and either can be the binding constraint.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        tokens_per_second: int | None = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = tokens_per_second
        self._request_allowance = float(requests_per_minute or 0)
        self._token_allowance = float(tokens_per_second or 0)
        self._last = time.monotonic()
        self._lock: asyncio.Lock | None = None
        self.waits = 0

    async def acquire(self, tokens: int = 0) -> None:
        if self.requests_per_minute is None and self.tokens_per_second is None:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                if self.requests_per_minute is not None:
                    self._request_allowance = min(
                        float(self.requests_per_minute),
                        self._request_allowance
                        + elapsed * self.requests_per_minute / 60.0,
                    )
                if self.tokens_per_second is not None:
                    self._token_allowance = min(
                        float(self.tokens_per_second),
                        self._token_allowance
                        + elapsed * self.tokens_per_second,
                    )

                request_ok = (
                    self.requests_per_minute is None
                    or self._request_allowance >= 1
                )
                token_ok = (
                    self.tokens_per_second is None
                    or self._token_allowance >= tokens
                )
                if request_ok and token_ok:
                    if self.requests_per_minute is not None:
                        self._request_allowance -= 1
                    if self.tokens_per_second is not None:
                        self._token_allowance -= tokens
                    return

                self.waits += 1
                await asyncio.sleep(0.01)


def estimate_tokens(state: Any) -> int:
    """Rough token count for a state, for rate-limiting purposes only.

    Four characters per token is the usual English approximation. It does not
    need to be exact -- it paces requests, it does not bill them.
    """
    return max(1, len(str(state)) // 4)


async def gather_decisions(
    states: Sequence[Any],
    decide: Callable[[Any], Awaitable[Mapping[str, Decision]]],
    limits: ProviderLimits,
    on_error: str = "raise",
) -> list[Mapping[str, Decision] | None]:
    """Answer every state concurrently, preserving input order."""
    semaphore = asyncio.Semaphore(max(1, limits.max_concurrency))
    bucket = TokenBucket(
        requests_per_minute=limits.requests_per_minute,
        tokens_per_second=limits.tokens_per_second,
    )

    async def one(state: Any) -> Mapping[str, Decision] | None:
        async with semaphore:
            await bucket.acquire(estimate_tokens(state))
            try:
                return await decide(state)
            except Exception:
                if on_error == "raise":
                    raise
                # "null" and "drop" both need the row marked as unanswered;
                # what happens to it afterwards is the caller's business.
                return None

    # ``gather`` preserves the order of its arguments regardless of completion
    # order, which is what keeps answers aligned with rows.
    return list(await asyncio.gather(*(one(state) for state in states)))


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run a coroutine from synchronous code.

    Parsers run inside ``schema.validate``, which may itself be called from
    inside a running event loop (a notebook, an async web handler). Creating a
    nested loop is not allowed, so in that case the work is handed to a worker
    thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
