"""Zilliz Cloud operations: collection, volume, and bulk import management."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from pymilvus import MilvusClient
from pymilvus.bulk_writer import bulk_import, get_import_progress
from pymilvus.bulk_writer.volume_manager import VolumeManager
from pymilvus.bulk_writer.volume_file_manager import VolumeFileManager

from .config import ZILLIZ_CLOUD_CN, PipelineConfig, build_schema

log = logging.getLogger(__name__)


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
