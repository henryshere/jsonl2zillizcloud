"""Core async pipeline: stream JSONL → embed → write Parquet → upload segments."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig, build_schema
from .embedder import create_embedder
from .writer import load_checkpoint, save_checkpoint, build_writer, make_row

log = logging.getLogger(__name__)


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
