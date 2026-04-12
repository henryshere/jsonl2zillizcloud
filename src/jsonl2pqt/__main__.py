"""
jsonl2pqt — JSONL → Zilliz Cloud with embeddings
-------------------------------------------------
End-to-end pipeline:
  1. (Optional) Create Zilliz Cloud collection with schema & indexes
  2. Stream JSONL → embed caption (remote API or local vLLM) → write Parquet
     Each segment is uploaded to Volume immediately after auto-flush,
     then the local file is deleted — local disk never exceeds --segment-size.
  3. Trigger bulk_import and poll until done

Usage:
  cd src && python -m jsonl2pqt --help
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from pymilvus import MilvusClient

from .config import PipelineConfig
from .cloud_ops import (
    create_collection,
    ensure_volume,
    create_volume_file_manager,
    run_bulk_import,
)
from .pipeline import stream_embed_write

log = logging.getLogger(__name__)


def run(args: argparse.Namespace):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        dim=args.dim,
        include_raw_json=not args.no_raw_json,
        include_caption_str=not args.no_caption_str,
        include_caption_json=not args.no_caption_json,
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
    p.add_argument("--no-caption-str",   action="store_true",
                                         help="Exclude caption_str field from schema and parquet output")
    p.add_argument("--no-caption-json",  action="store_true",
                                         help="Exclude caption_json field from schema and parquet output (also skips JSON parsing)")
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
