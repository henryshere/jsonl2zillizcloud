"""
jsonl_to_parquet.py
-------------------
End-to-end pipeline:
  1. (Optional) Create Zilliz Cloud collection with schema & indexes
  2. Stream JSONL → embed caption (remote API or local vLLM) → write Parquet
     Each segment is uploaded to Volume immediately after auto-flush,
     then the local file is deleted — local disk never exceeds --segment-size.
  3. Trigger bulk_import and poll until done

Install:
  pip install "pymilvus[bulk_writer]>=2.5" tqdm
  pip install aiohttp   # required for --embed-mode=api only
  pip install vllm      # required for --embed-mode=local only

Usage:
  export ZILLIZ_API_KEY=xxx

  # Local mode (vLLM, requires GPU, no API key needed):
  python jsonl_to_parquet.py \
      --input            data.jsonl \
      --output-dir       ./bulk_output \
      --collection       caption_collection \
      --cluster-endpoint https://xxx.api.ali-cn-beijing.cloud.zilliz.com.cn \
      --cluster-token    xxx \
      --cluster-id       inxx-xxxx \
      --project-id       proj-xxxx \
      --embed-mode       local \
      --model            Qwen3-Embedding-4B \
      --dim              512 \
      --batch-size       256

  # API mode (remote, any OpenAI-compatible endpoint):
  export EMBED_API_KEY=sk-xxx
  python jsonl_to_parquet.py \
      --input            data.jsonl \
      --output-dir       ./bulk_output \
      --collection       caption_collection \
      --cluster-endpoint https://xxx.api.ali-cn-beijing.cloud.zilliz.com.cn \
      --cluster-token    xxx \
      --cluster-id       inxx-xxxx \
      --project-id       proj-xxxx \
      --embed-mode       api \
      --api-base         https://api.siliconflow.cn/v1/embeddings \
      --model            Qwen3-Embedding-4B \
      --dim              512 \
      --batch-size       32 \
      --workers          8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import (
    LocalBulkWriter,
    BulkFileType,
    bulk_import,
    get_import_progress,
)
from pymilvus.bulk_writer.volume_manager import VolumeManager
from pymilvus.bulk_writer.volume_file_manager import VolumeFileManager

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ZILLIZ_CLOUD_CN = "https://api.cloud.zilliz.com.cn"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Centralized pipeline settings, built once from CLI args."""
    dim: int = 512
    include_raw_json: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Schema (shared by collection creation and bulk writer)
# ──────────────────────────────────────────────────────────────────────────────
def build_schema(config: PipelineConfig):
    """Build the collection/writer schema. Field count depends on config flags."""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("autoid",                 DataType.INT64,        is_primary=True)
    schema.add_field("id",                     DataType.VARCHAR,      max_length=128)
    schema.add_field("path",                   DataType.VARCHAR,      max_length=1024)
    schema.add_field("height",                 DataType.INT32)
    schema.add_field("width",                  DataType.INT32)
    schema.add_field("caption",                DataType.VARCHAR,      max_length=65535)
    schema.add_field("caption_json",           DataType.JSON)
    schema.add_field("caption_version",        DataType.VARCHAR,      max_length=32)
    schema.add_field("text_ratio",             DataType.FLOAT)
    schema.add_field("craft_bbox_num",         DataType.INT32)
    schema.add_field("fused_image",            DataType.FLOAT)
    schema.add_field("fused_image_aesthetic",  DataType.FLOAT)
    schema.add_field("fused_image_technical",  DataType.FLOAT)
    schema.add_field("image_512",              DataType.VARCHAR,      max_length=1024)
    schema.add_field("rand",                   DataType.FLOAT)
    if config.include_raw_json:
        schema.add_field("raw_json",               DataType.JSON)
    schema.add_field("caption_vector",         DataType.FLOAT_VECTOR, dim=config.dim)
    schema.verify()

    return schema


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Create collection
# ──────────────────────────────────────────────────────────────────────────────
def create_collection(client: MilvusClient, name: str, config: PipelineConfig) -> None:
    """Create collection with auto_id, schema, and indexes."""
    if client.has_collection(name):
        log.info("Collection '%s' already exists, skipping creation.", name)
        return

    schema = build_schema(config)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="caption_vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="height",          index_type="AUTOINDEX")
    index_params.add_index(field_name="width",           index_type="AUTOINDEX")
    index_params.add_index(field_name="caption_version", index_type="AUTOINDEX")

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    log.info("Collection '%s' created.", name)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2a: Embedding — remote API (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Step 2a (alt): Embedding — local vLLM inference
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Embedder factory
# ──────────────────────────────────────────────────────────────────────────────
def create_embedder(args) -> ApiEmbedder | LocalEmbedder:
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


# ──────────────────────────────────────────────────────────────────────────────
# Step 2b: Stream JSONL → embed → write Parquet
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: Path) -> tuple[int, int]:
    """Return (line_offset, segment_idx) to resume from."""
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
            offset = data.get("line_offset", 0)
            seg_idx = data.get("segment_idx", 0)
            log.info("Resuming from checkpoint: line %d, segment %d", offset, seg_idx)
            return offset, seg_idx
    return 0, 0


def save_checkpoint(ckpt_path: Path, line_offset: int, segment_idx: int) -> None:
    with open(ckpt_path, "w") as f:
        json.dump({"line_offset": line_offset, "segment_idx": segment_idx}, f)


def build_writer(schema, output_dir: str, segment_size: int) -> LocalBulkWriter:
    return LocalBulkWriter(
        schema=schema,
        local_path=output_dir,
        chunk_size=segment_size,
        file_type=BulkFileType.PARQUET,
    )


def make_row(record: dict, vector: list[float] | None, config: PipelineConfig) -> dict:
    """
    Build a row dict for LocalBulkWriter.append_row().
    Includes all 13 original fields + caption_json + caption_vector.
    Optionally includes raw_json (controlled by config.include_raw_json).
    Does NOT include autoid (Zilliz auto-generates it).
    """
    caption_str = str(record.get("caption", ""))
    try:
        caption_json = json.loads(caption_str) if caption_str else {}
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("caption is not valid JSON for id=%s: %s", record.get("id", "?"), exc)
        caption_json = {"_error": f"JSONDecodeError: {exc}"}

    # Type guard: must be a dict
    if not isinstance(caption_json, dict):
        log.warning("caption parsed to %s (not dict) for id=%s", type(caption_json).__name__, record.get("id", "?"))
        caption_json = {"_error": f"expected dict, got {type(caption_json).__name__}"}

    # Key sanitization: replace spaces and special chars with _
    caption_json = {
        re.sub(r'[^a-zA-Z0-9_]', '_', k): v
        for k, v in caption_json.items()
    }

    # Size guard: must fit in 64 KB
    if len(json.dumps(caption_json, ensure_ascii=False).encode("utf-8")) > 65536:
        log.warning("caption_json exceeds 64 KB for id=%s", record.get("id", "?"))
        caption_json = {"_error": "caption_json exceeds 64 KB"}

    row = {
        "id":                     str(record.get("id", "")),
        "path":                   str(record.get("path", "")),
        "height":                 int(record.get("height", 0)),
        "width":                  int(record.get("width", 0)),
        "caption":                caption_str,
        "caption_json":           caption_json,
        "caption_version":        str(record.get("caption_version", "")),
        "text_ratio":             float(record.get("text_ratio", 0.0)),
        "craft_bbox_num":         int(record.get("craft_bbox_num", 0)),
        "fused_image":            float(record.get("fused_image", 0.0)),
        "fused_image_aesthetic":  float(record.get("fused_image_aesthetic", 0.0)),
        "fused_image_technical":  float(record.get("fused_image_technical", 0.0)),
        "image_512":              str(record.get("image_512", "")),
        "rand":                   float(record.get("rand", 0.0)),
        "caption_vector":         [float(x) for x in vector] if vector else [0.0] * config.dim,
    }

    if config.include_raw_json:
        raw_json_str = json.dumps(record, ensure_ascii=False)
        if len(raw_json_str.encode("utf-8")) > 65536:
            log.warning("raw_json exceeds 64 KB for id=%s, truncating.", record.get("id", "?"))
            raw_json_str = raw_json_str[:65000] + "…}"
        row["raw_json"] = json.loads(raw_json_str)

    return row


async def stream_embed_write(args: argparse.Namespace, vfm, config: PipelineConfig) -> None:
    """
    Main processing loop.
    Streams JSONL → embeds → writes Parquet segments.
    Each segment is uploaded to Volume and deleted locally immediately.
    """
    schema = build_schema(config)

    ckpt_path = Path(args.output_dir) / "checkpoint.json"
    resume_offset, resumed_segment_idx = load_checkpoint(ckpt_path)

    embedder = create_embedder(args)

    log.info("Segment size: %d MB. Writer auto-flushes at this size.",
             args.segment_size // (1024 * 1024))

    # State
    writer = build_writer(schema, args.output_dir, args.segment_size)
    total_rows = 0
    total_uploaded = 0

    # Accumulators for micro-batching
    batch_texts: list[str] = []        # texts to embed ("" if no caption)
    batch_records: list[dict] = []     # corresponding records
    texts_in_batch = 0
    segment_idx = resumed_segment_idx
    line_no = 0

    async def upload_ready_parquets(label: str = ""):
        """Scan output dir for .parquet files, upload and delete them."""
        nonlocal segment_idx, total_uploaded
        data_dir = Path(str(writer.data_path))
        if not data_dir.exists():
            return
        parquet_files = sorted(data_dir.glob("*.parquet"))
        for pf in parquet_files:
            remote_name = f"part-{segment_idx:06d}.parquet"
            renamed = pf.parent / remote_name
            pf.rename(renamed)

            uploaded = False
            for attempt in range(1, 4):
                try:
                    await asyncio.to_thread(
                        vfm.upload_file_to_volume,
                        source_file_path=str(renamed),
                        target_volume_path="data/",
                    )
                    uploaded = True
                    break
                except Exception as exc:
                    wait = min(2 ** attempt, 30)
                    log.warning("Upload failed (attempt %d/3): %s, retry in %ds",
                                attempt, exc, wait)
                    await asyncio.sleep(wait)

            tag = f"[{label}] " if label else ""
            if uploaded:
                log.info("%sUploaded %s (%d MB)", tag, remote_name,
                         renamed.stat().st_size // (1024 * 1024))
                renamed.unlink()
                segment_idx += 1
                total_uploaded += 1
                save_checkpoint(ckpt_path, line_no, segment_idx)
            else:
                log.error("%sFailed to upload %s after 3 attempts. "
                          "File kept at %s for manual retry.", tag, remote_name, renamed)
                raise RuntimeError(f"Upload failed for {remote_name}")

    async def flush_batch():
        """Embed accumulated texts and write all rows to writer."""
        nonlocal texts_in_batch
        if not batch_records:
            return

        # Collect texts that need embedding
        texts_to_embed = [t for t in batch_texts if t]

        vectors = []
        if texts_to_embed:
            # Split into sub-batches sized by --batch-size
            bs = args.batch_size
            sub_batches = [
                texts_to_embed[i:i + bs]
                for i in range(0, len(texts_to_embed), bs)
            ]
            results = await embedder.embed_batches(sub_batches)
            for sub_result in results:
                vectors.extend(sub_result)

        vec_idx = 0
        for record, text in zip(batch_records, batch_texts):
            if text:
                vec = vectors[vec_idx]
                vec_idx += 1
            else:
                vec = None
            row = make_row(record, vec, config)
            writer.append_row(row)

        batch_texts.clear()
        batch_records.clear()
        texts_in_batch = 0

        # Check if writer auto-flushed any Parquet files
        await upload_ready_parquets("auto-flush")

    with tqdm(desc="Processing", unit="row") as bar:
        if resume_offset:
            bar.update(resume_offset)
        with open(args.input, encoding="utf-8") as fh:
            for raw_line in fh:
                line_no += 1
                if line_no <= resume_offset:
                    continue

                raw_line = raw_line.strip()
                if not raw_line:
                    bar.update(1)
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed line %d: %s", line_no, exc)
                    bar.update(1)
                    continue

                caption = record.get("caption", "")
                has_text = bool(caption and str(caption).strip())

                batch_records.append(record)
                batch_texts.append(str(caption) if has_text else "")

                if has_text:
                    texts_in_batch += 1
                else:
                    log.warning("Empty caption at line %d, id=%s", line_no, record.get("id", "?"))

                # Flush embedding batch when full
                if texts_in_batch >= args.batch_size:
                    await flush_batch()

                # Save checkpoint periodically (every 10k lines)
                if line_no % 10000 == 0:
                    save_checkpoint(ckpt_path, line_no, segment_idx)

                total_rows += 1
                bar.update(1)

    # Final flushes
    await flush_batch()
    writer.commit()
    await upload_ready_parquets("final")
    await embedder.close()

    save_checkpoint(ckpt_path, line_no, segment_idx)
    log.info("All done. %d rows processed, %d segment(s) uploaded.", total_rows, total_uploaded)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Ensure Volume exists
# ──────────────────────────────────────────────────────────────────────────────
def ensure_volume(api_key: str, project_id: str, region_id: str, volume_name: str):
    """Create volume if it doesn't exist."""
    vm = VolumeManager(cloud_endpoint=ZILLIZ_CLOUD_CN, api_key=api_key)

    # Check existing volumes
    resp = vm.list_volumes(project_id=project_id)
    data = resp.json() if hasattr(resp, "json") else resp
    volumes = data.get("data", {}).get("volumes", []) if isinstance(data, dict) else []
    for v in volumes:
        if v.get("volumeName") == volume_name:
            log.info("Volume '%s' already exists.", volume_name)
            return

    vm.create_volume(
        project_id=project_id,
        region_id=region_id,
        volume_name=volume_name,
    )
    log.info("Volume '%s' created.", volume_name)


def create_volume_file_manager(api_key: str, volume_name: str) -> VolumeFileManager:
    """Create a VolumeFileManager for uploading files."""
    return VolumeFileManager(
        cloud_endpoint=ZILLIZ_CLOUD_CN,
        api_key=api_key,
        volume_name=volume_name,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Trigger bulk_import and poll progress
# ──────────────────────────────────────────────────────────────────────────────
def run_bulk_import(args: argparse.Namespace):
    """Call bulk_import and poll until complete."""
    resp = bulk_import(
        url=ZILLIZ_CLOUD_CN,
        api_key=args.zilliz_api_key,
        cluster_id=args.cluster_id,
        collection_name=args.collection,
        volume_name=args.volume_name,
        data_paths=[["data/"]],
    )
    resp_data = resp.json()
    job_id = resp_data.get("data", {}).get("jobId")
    if not job_id:
        log.error("bulk_import failed: %s", resp_data)
        sys.exit(1)

    log.info("bulk_import job started: %s", job_id)

    while True:
        progress_resp = get_import_progress(
            url=ZILLIZ_CLOUD_CN,
            api_key=args.zilliz_api_key,
            cluster_id=args.cluster_id,
            job_id=job_id,
        )
        progress_data = progress_resp.json()
        state = progress_data.get("data", {}).get("state", "Unknown")
        progress = progress_data.get("data", {}).get("progress", 0)
        log.info("Import progress: %s — %d%%", state, progress)

        if state in ("Completed", "Failed"):
            break
        time.sleep(10)

    if state == "Failed":
        log.error("Import job %s FAILED: %s", job_id, progress_data)
        sys.exit(1)

    log.info("Import job %s completed successfully.", job_id)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        dim=args.dim,
        include_raw_json=not args.no_raw_json,
    )

    # Step 1: Create collection (unless --skip-create)
    if not args.skip_create:
        client = MilvusClient(uri=args.cluster_endpoint, token=args.cluster_token)
        create_collection(client, args.collection, config)
        client.close()

    # Step 1b: Ensure Volume exists
    ensure_volume(args.zilliz_api_key, args.project_id, args.region_id, args.volume_name)
    vfm = create_volume_file_manager(args.zilliz_api_key, args.volume_name)

    # Step 2: Stream → embed → write Parquet → upload each segment → delete local
    asyncio.run(stream_embed_write(args, vfm, config))

    # Step 3: bulk_import (all segments are already on Volume)
    run_bulk_import(args)

    log.info("All done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JSONL → Zilliz Cloud with embeddings (API or local vLLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",        required=True, help="Input JSONL file path")
    p.add_argument("--output-dir",   default="./bulk_output", help="Local dir for Parquet output")

    # Embedding
    p.add_argument("--embed-mode",   choices=["api", "local"], default="api",
                                     help="'api' for remote API, 'local' for vLLM local inference")
    p.add_argument("--model",        default="Qwen3-Embedding-4B", help="Embedding model name")
    p.add_argument("--dim",          type=int, default=512, help="Embedding dimension")
    p.add_argument("--batch-size",   type=int, default=256,  help="Texts per embed call (api: max 32; local: higher for GPU)")
    p.add_argument("--workers",      type=int, default=8,    help="Concurrent API workers (ignored in local mode)")
    p.add_argument("--max-retries",  type=int, default=3,    help="Max retries per API call (ignored in local mode)")
    p.add_argument("--api-base",     default="",
                                     help="Embedding API endpoint URL (required in api mode)")
    p.add_argument("--api-key",      default=os.getenv("EMBED_API_KEY", ""),
                                     help="Embedding API key (api mode only)")

    # Writer
    p.add_argument("--segment-size", type=int, default=1024*1024*128, help="Bytes per Parquet chunk")

    # Zilliz Cloud
    p.add_argument("--collection",       required=True, help="Collection name")
    p.add_argument("--cluster-endpoint", default=os.getenv("ZILLIZ_ENDPOINT", ""), help="Cluster endpoint URL")
    p.add_argument("--cluster-token",    default=os.getenv("ZILLIZ_TOKEN", ""),     help="Cluster access token")
    p.add_argument("--cluster-id",       default=os.getenv("ZILLIZ_CLUSTER_ID", ""), help="Cluster ID")
    p.add_argument("--zilliz-api-key",   default=os.getenv("ZILLIZ_API_KEY", ""),    help="Zilliz Cloud API key")
    p.add_argument("--volume-name",      default=os.getenv("ZILLIZ_VOLUME", "bulk_import_vol"), help="Volume name")
    p.add_argument("--project-id",       default=os.getenv("ZILLIZ_PROJECT_ID", ""), help="Project ID")
    p.add_argument("--region-id",        default=os.getenv("ZILLIZ_REGION_ID", "ali-cn-beijing"), help="Region ID")
    p.add_argument("--skip-create",      action="store_true", help="Skip collection creation")
    p.add_argument("--no-raw-json",      action="store_true", help="Exclude raw_json field from schema and parquet output")

    args = p.parse_args()

    # Validate required keys
    if args.embed_mode == "api" and not args.api_key:
        p.error("--api-key or EMBED_API_KEY env var required when --embed-mode=api")
    if args.embed_mode == "api" and not args.api_base:
        p.error("--api-base required when --embed-mode=api")
    if not args.skip_create and (not args.cluster_endpoint or not args.cluster_token):
        p.error("--cluster-endpoint and --cluster-token required unless --skip-create")
    if not args.zilliz_api_key:
        p.error("Zilliz Cloud API key required (--zilliz-api-key or ZILLIZ_API_KEY env var)")
    if not args.cluster_id:
        p.error("--cluster-id required (or ZILLIZ_CLUSTER_ID env var)")
    if not args.project_id:
        p.error("--project-id required (or ZILLIZ_PROJECT_ID env var)")

    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run(parse_args())
