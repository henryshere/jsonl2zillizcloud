"""Checkpoint management, LocalBulkWriter factory, and row builder."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType

from .config import PipelineConfig

log = logging.getLogger(__name__)


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
    Always includes the base fields + caption_vector.
    Optionally includes caption_str, caption_json, raw_json based on config flags.
    Does NOT include autoid (Zilliz auto-generates it).
    """
    # caption_str: only materialized if stored
    caption_str = str(record.get("caption", "")) if config.include_caption_str else None

    # caption_json: parsing + sanitization + size guard skipped entirely when excluded
    caption_json = None
    if config.include_caption_json:
        raw_caption = str(record.get("caption", ""))
        try:
            caption_json = json.loads(raw_caption) if raw_caption else {}
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("caption is not valid JSON for id=%s: %s", record.get("id", "?"), exc)
            caption_json = {"_error": f"JSONDecodeError: {exc}"}

        # Type guard: must be a dict
        if not isinstance(caption_json, dict):
            log.warning("caption parsed to %s (not dict) for id=%s",
                        type(caption_json).__name__, record.get("id", "?"))
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

    if config.include_caption_str:
        row["caption_str"] = caption_str
    if config.include_caption_json:
        row["caption_json"] = caption_json
    if config.include_raw_json:
        raw_json_str = json.dumps(record, ensure_ascii=False)
        if len(raw_json_str.encode("utf-8")) > 65536:
            log.warning("raw_json exceeds 64 KB for id=%s, truncating.", record.get("id", "?"))
            raw_json_str = raw_json_str[:65000] + "…}"
        row["raw_json"] = json.loads(raw_json_str)

    return row
