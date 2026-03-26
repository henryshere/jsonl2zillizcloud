"""
jsonl_to_parquet.py
-------------------
End-to-end pipeline:
  1. (Optional) Create Zilliz Cloud collection with schema & indexes
  2. Stream JSONL → embed caption via SiliconFlow API → write Parquet
     Each segment is uploaded to Volume immediately after flush,
     then the local file is deleted — local disk never exceeds ~1 GB.
  3. Trigger bulk_import and poll until done

Install:
  pip install "pymilvus[bulk_writer]>=2.5" aiohttp tqdm tenacity

Usage:
  export SILICONFLOW_API_KEY=sk-xxx
  export ZILLIZ_API_KEY=xxx

  python jsonl_to_parquet.py \
      --input            data.jsonl \
      --output-dir       ./bulk_output \
      --collection       caption_collection \
      --cluster-endpoint https://xxx.api.ali-cn-beijing.cloud.zilliz.com.cn \
      --cluster-token    xxx \
      --cluster-id       inxx-xxxx \
      --project-id       proj-xxxx \
      --model            BAAI/bge-m3 \
      --dim              1024 \
      --batch-size       32 \
      --workers          8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SF_ENDPOINT = "https://api.siliconflow.cn/v1/embeddings"
ZILLIZ_CLOUD_CN = "https://api.cloud.zilliz.com.cn"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Create collection
# ──────────────────────────────────────────────────────────────────────────────
def create_collection(client: MilvusClient, name: str, dim: int) -> None:
    """Create collection with auto_id, schema, and indexes."""
    if client.has_collection(name):
        log.info("Collection '%s' already exists, skipping creation.", name)
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("autoid",                 DataType.INT64,        is_primary=True)
    schema.add_field("id",                     DataType.VARCHAR,      max_length=128)
    schema.add_field("path",                   DataType.VARCHAR,      max_length=1024)
    schema.add_field("height",                 DataType.INT32)
    schema.add_field("width",                  DataType.INT32)
    schema.add_field("caption",                DataType.VARCHAR,      max_length=65535)
    schema.add_field("caption_version",        DataType.VARCHAR,      max_length=32)
    schema.add_field("text_ratio",             DataType.FLOAT)
    schema.add_field("craft_bbox_num",         DataType.INT32)
    schema.add_field("fused_image",            DataType.FLOAT)
    schema.add_field("fused_image_aesthetic",  DataType.FLOAT)
    schema.add_field("fused_image_technical",  DataType.FLOAT)
    schema.add_field("image_512",              DataType.VARCHAR,      max_length=1024)
    schema.add_field("rand",                   DataType.FLOAT)
    schema.add_field("raw_json",               DataType.JSON)
    schema.add_field("caption_vector",         DataType.FLOAT_VECTOR, dim=dim)

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
# Step 2a: SiliconFlow embedding (async, concurrent workers)
# ──────────────────────────────────────────────────────────────────────────────
class SiliconFlowEmbedder:
    """Async client that calls SiliconFlow /v1/embeddings with concurrency."""

    def __init__(self, api_key: str, model: str, dim: int,
                 workers: int, max_retries: int):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.workers = workers
        self.max_retries = max_retries
        self._sem: asyncio.Semaphore | None = None
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            # import ssl
            # ssl_ctx = ssl.create_default_context()
            # ssl_ctx.check_hostname = False
            # ssl_ctx.verify_mode = ssl.CERT_NONE
            # connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                # connector=connector,
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
            async with self._sem:
                try:
                    async with self._session.post(SF_ENDPOINT, json=payload) as resp:
                        if resp.status == 429:
                            wait = min(2 ** attempt, 30)
                            log.warning("Rate limited (429), waiting %ds …", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        body = await resp.json()
                        # sort by index to guarantee order
                        data = sorted(body["data"], key=lambda d: d["index"])
                        return [d["embedding"] for d in data]
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    wait = min(2 ** attempt, 30)
                    log.warning("API error (attempt %d/%d): %s, retry in %ds",
                                attempt, self.max_retries, exc, wait)
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"SiliconFlow API failed after {self.max_retries} retries: {last_exc}"
        )

    async def embed_batches(self, batches: list[list[str]]) -> list[list[list[float]]]:
        """Embed multiple batches concurrently."""
        tasks = [self._call_api(b) for b in batches]
        return await asyncio.gather(*tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2b: Stream JSONL → embed → write Parquet
# ──────────────────────────────────────────────────────────────────────────────
def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def load_checkpoint(ckpt_path: Path) -> int:
    """Return the line offset to resume from (0-based)."""
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
            offset = data.get("line_offset", 0)
            log.info("Resuming from checkpoint: line %d", offset)
            return offset
    return 0


def save_checkpoint(ckpt_path: Path, line_offset: int) -> None:
    with open(ckpt_path, "w") as f:
        json.dump({"line_offset": line_offset}, f)


def build_writer(schema, output_dir: str, segment_size: int) -> LocalBulkWriter:
    return LocalBulkWriter(
        schema=schema,
        local_path=output_dir,
        chunk_size=segment_size,
        file_type=BulkFileType.PARQUET,
    )



def build_schema_for_writer(dim: int):
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("autoid",                 DataType.INT64,        is_primary=True)
    schema.add_field("id",                     DataType.VARCHAR,      max_length=128)
    schema.add_field("path",                   DataType.VARCHAR,      max_length=1024)
    schema.add_field("height",                 DataType.INT32)
    schema.add_field("width",                  DataType.INT32)
    schema.add_field("caption",                DataType.VARCHAR,      max_length=65535)
    schema.add_field("caption_version",        DataType.VARCHAR,      max_length=32)
    schema.add_field("text_ratio",             DataType.FLOAT)
    schema.add_field("craft_bbox_num",         DataType.INT32)
    schema.add_field("fused_image",            DataType.FLOAT)
    schema.add_field("fused_image_aesthetic",  DataType.FLOAT)
    schema.add_field("fused_image_technical",  DataType.FLOAT)
    schema.add_field("image_512",              DataType.VARCHAR,      max_length=1024)
    schema.add_field("rand",                   DataType.FLOAT)
    schema.add_field("raw_json",               DataType.JSON)
    schema.add_field("caption_vector",         DataType.FLOAT_VECTOR, dim=dim)
    schema.verify()

    return schema


def make_row(record: dict, vector: list[float] | None, dim: int) -> dict:
    """
    Build a row dict for LocalBulkWriter.append_row().
    Includes all 13 original fields + raw_json + caption_vector.
    Does NOT include autoid (Zilliz auto-generates it).
    """
    raw_json_str = json.dumps(record, ensure_ascii=False)
    # guard: JSON field max 64 KB
    if len(raw_json_str.encode("utf-8")) > 65536:
        log.warning("raw_json exceeds 64 KB for id=%s, truncating.", record.get("id", "?"))
        raw_json_str = raw_json_str[:65000] + "…}"
    return {
        "id":                     str(record.get("id", "")),
        "path":                   str(record.get("path", "")),
        "height":                 int(record.get("height", 0)),
        "width":                  int(record.get("width", 0)),
        "caption":                str(record.get("caption", "")),
        "caption_version":        str(record.get("caption_version", "")),
        "text_ratio":             float(record.get("text_ratio", 0.0)),
        "craft_bbox_num":         int(record.get("craft_bbox_num", 0)),
        "fused_image":            float(record.get("fused_image", 0.0)),
        "fused_image_aesthetic":  float(record.get("fused_image_aesthetic", 0.0)),
        "fused_image_technical":  float(record.get("fused_image_technical", 0.0)),
        "image_512":              str(record.get("image_512", "")),
        "rand":                   float(record.get("rand", 0.0)),
        "raw_json":               json.loads(raw_json_str),  # JSON field expects dict
        "caption_vector":         [float(x) for x in vector] if vector else [0.0] * dim,
    }


async def stream_embed_write(args: argparse.Namespace, vfm) -> None:
    """
    Main processing loop.
    Streams JSONL → embeds → writes Parquet segments.
    Each segment is uploaded to Volume and deleted locally immediately.
    """
    schema = build_schema_for_writer(args.dim)

    ckpt_path = Path(args.output_dir) / "checkpoint.json"
    resume_offset = load_checkpoint(ckpt_path)

    embedder = SiliconFlowEmbedder(
        api_key=args.sf_api_key,
        model=args.model,
        dim=args.dim,
        workers=args.workers,
        max_retries=args.max_retries,
    )

    log.info("Counting lines in %s …", args.input)
    total = count_lines(args.input)
    log.info("~%d lines found.", total)

    log.info("Segment size: %d MB. Writer auto-flushes at this size.",
             args.segment_size // (1024 * 1024))

    # State
    writer = build_writer(schema, args.output_dir, args.segment_size)
    total_rows = 0
    total_uploaded = 0

    # Accumulators for micro-batching
    batch_texts: list[str] = []        # texts to embed
    batch_records: list[dict] = []     # corresponding records
    batch_has_text: list[bool] = []    # whether each record has text

    async def flush_batch():
        """Embed accumulated texts and write all rows to writer."""
        if not batch_records:
            return

        # Collect texts that need embedding
        texts_to_embed = [t for t, has in zip(batch_texts, batch_has_text) if has]

        vectors = []
        if texts_to_embed:
            # Split into API-sized sub-batches (max 32)
            sub_batches = [
                texts_to_embed[i:i + 32]
                for i in range(0, len(texts_to_embed), 32)
            ]
            results = await embedder.embed_batches(sub_batches)
            for sub_result in results:
                vectors.extend(sub_result)

        vec_idx = 0
        for record, has_text in zip(batch_records, batch_has_text):
            if has_text:
                vec = vectors[vec_idx]
                vec_idx += 1
            else:
                vec = None
            row = make_row(record, vec, args.dim)
            writer.append_row(row)

        batch_texts.clear()
        batch_records.clear()
        batch_has_text.clear()

        # Check if writer auto-flushed any Parquet files
        upload_ready_parquets("auto-flush")

    segment_idx = 0

    def upload_ready_parquets(label: str = ""):
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
            vfm.upload_file_to_volume(
                source_file_path=str(renamed),
                target_volume_path="data/",
            )
            tag = f"[{label}] " if label else ""
            log.info("%sUploaded %s (%d MB)", tag, remote_name,
                     renamed.stat().st_size // (1024 * 1024))
            renamed.unlink()
            segment_idx += 1
            total_uploaded += 1

    line_no = 0
    with tqdm(total=total, desc="Processing", unit="row", initial=resume_offset) as bar:
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
                batch_has_text.append(has_text)

                if not has_text:
                    log.warning("Empty caption at line %d, id=%s", line_no, record.get("id", "?"))

                # Flush embedding batch when full
                if sum(batch_has_text) >= args.batch_size:
                    await flush_batch()

                # Save checkpoint periodically (every 10k lines)
                if line_no % 10000 == 0:
                    save_checkpoint(ckpt_path, line_no)

                total_rows += 1
                bar.update(1)

    # Final flushes
    await flush_batch()
    writer.commit()
    upload_ready_parquets("final")
    await embedder.close()

    save_checkpoint(ckpt_path, line_no)
    log.info("All done. %d rows processed, %d segment(s) uploaded.", total_rows, total_uploaded)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Upload to Volume
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
# Step 4: Trigger bulk_import and poll progress
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

    # Step 1: Create collection (unless --skip-create)
    if not args.skip_create:
        client = MilvusClient(uri=args.cluster_endpoint, token=args.cluster_token)
        create_collection(client, args.collection, args.dim)
        client.close()

    # Step 2: Ensure Volume exists
    ensure_volume(args.zilliz_api_key, args.project_id, args.region_id, args.volume_name)
    vfm = create_volume_file_manager(args.zilliz_api_key, args.volume_name)

    # Step 3: Stream → embed → write Parquet → upload each segment → delete local
    asyncio.run(stream_embed_write(args, vfm))

    # Step 4: bulk_import (all segments are already on Volume)
    run_bulk_import(args)

    log.info("All done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JSONL → Zilliz Cloud with SiliconFlow embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",        required=True, help="Input JSONL file path")
    p.add_argument("--output-dir",   default="./bulk_output", help="Local dir for Parquet output")

    # Embedding
    p.add_argument("--model",        default="BAAI/bge-m3", help="SiliconFlow model name")
    p.add_argument("--dim",          type=int, default=1024, help="Embedding dimension")
    p.add_argument("--batch-size",   type=int, default=32,   help="Texts per API call (max 32)")
    p.add_argument("--workers",      type=int, default=8,    help="Concurrent API workers")
    p.add_argument("--max-retries",  type=int, default=3,    help="Max retries per API call")

    # Writer
    p.add_argument("--segment-size", type=int, default=1024*1024*10, help="Bytes per Parquet chunk")

    # Zilliz Cloud
    p.add_argument("--collection",       required=True, help="Collection name")
    p.add_argument("--cluster-endpoint", default=os.getenv("ZILLIZ_ENDPOINT", ""), help="Cluster endpoint URL")
    p.add_argument("--cluster-token",    default=os.getenv("ZILLIZ_TOKEN", ""),     help="Cluster access token")
    p.add_argument("--cluster-id",       default=os.getenv("ZILLIZ_CLUSTER_ID", ""), help="Cluster ID")
    p.add_argument("--zilliz-api-key",   default=os.getenv("ZILLIZ_API_KEY", ""),    help="Zilliz Cloud API key")
    p.add_argument("--volume-name",      default="bulk_import_vol",                  help="Volume name")
    p.add_argument("--project-id",       default=os.getenv("ZILLIZ_PROJECT_ID", ""), help="Project ID")
    p.add_argument("--region-id",        default="ali-cn-beijing",                   help="Region ID")
    p.add_argument("--skip-create",      action="store_true", help="Skip collection creation")

    # SiliconFlow
    p.add_argument("--sf-api-key", default=os.getenv("SILICONFLOW_API_KEY", ""), help="SiliconFlow API key")

    args = p.parse_args()

    # Validate required keys
    if not args.sf_api_key:
        p.error("SiliconFlow API key required (--sf-api-key or SILICONFLOW_API_KEY env var)")
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
    run(parse_args())
