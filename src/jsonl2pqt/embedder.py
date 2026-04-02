"""Embedding backends: remote API (OpenAI-compatible) and local vLLM."""

from __future__ import annotations

import asyncio
import logging
import sys

log = logging.getLogger(__name__)


class _RateLimited(Exception):
    """Sentinel for 429 responses — used to break out of semaphore before sleeping."""
    pass


class ApiEmbedder:
    """Async client that calls any OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, api_key: str, model: str, dim: int,
                 workers: int, max_retries: int, endpoint: str):
        import aiohttp as _aiohttp
        self._aiohttp = _aiohttp
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.workers = workers
        self.max_retries = max_retries
        self.endpoint = endpoint
        self._sem: asyncio.Semaphore | None = None
        self._session = None

    async def _ensure_session(self):
        aiohttp = self._aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=120),
            )
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.workers)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Single API call with retry."""
        await self._ensure_session()
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._sem:
                    async with self._session.post(self.endpoint, json=payload) as resp:
                        if resp.status == 429:
                            raise _RateLimited()
                        resp.raise_for_status()
                        body = await resp.json()
                        # sort by index to guarantee order
                        data = sorted(body["data"], key=lambda d: d["index"])
                        return [d["embedding"] for d in data]
            except (_RateLimited, self._aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                log.warning("API error (attempt %d/%d): %s, retry in %ds",
                            attempt, self.max_retries, exc, wait)
                await asyncio.sleep(wait)  # semaphore already released

        raise RuntimeError(
            f"Embedding API failed after {self.max_retries} retries: {last_exc}"
        )

    async def embed_batches(self, batches: list[list[str]]) -> list[list[list[float]]]:
        """Embed multiple batches concurrently."""
        tasks = [self._call_api(b) for b in batches]
        return await asyncio.gather(*tasks)


class LocalEmbedder:
    """Local embedding using vLLM's offline LLM engine.
    --workers is ignored; vLLM manages GPU parallelism internally.
    --batch-size controls how many texts are fed to llm.embed() at once."""

    def __init__(self, model: str, dim: int):
        try:
            from vllm import LLM
        except ImportError:
            sys.exit("Run:  pip install vllm")
        log.info("Loading local model '%s' via vLLM …", model)
        self.model_name = model
        self.dim = dim
        self.llm = LLM(model=model, runner="pooling")

    async def embed_batches(self, batches: list[list[str]]) -> list[list[list[float]]]:
        """Embed batches locally. No sub-batching — vLLM handles parallelism."""
        results = []
        for batch in batches:
            outputs = await asyncio.to_thread(self.llm.embed, batch)
            results.append([o.outputs.embedding[:self.dim] for o in outputs])
        return results

    async def close(self):
        pass  # no-op for local


def create_embedder(args) -> ApiEmbedder | LocalEmbedder:
    """Factory: return the right embedder based on --embed-mode."""
    if args.embed_mode == "local":
        return LocalEmbedder(model=args.model, dim=args.dim)
    else:
        if not args.api_key:
            raise ValueError("--api-key or EMBED_API_KEY env var required when --embed-mode=api")
        return ApiEmbedder(
            api_key=args.api_key, model=args.model, dim=args.dim,
            workers=args.workers, max_retries=args.max_retries,
            endpoint=args.api_base,
        )
