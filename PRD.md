# PRD: JSONL to Zilliz Cloud with Embedding Enrichment

## Overview

A Python CLI script that:

1. **Creates** the Zilliz Cloud collection (with schema, indexes, and `auto_id=True`)
2. **Ensures** the Zilliz Cloud Volume exists (Alibaba Cloud, Beijing region)
3. **Streams** a massive JSONL file (up to 500M rows) in constant memory, embeds the `caption` field via remote API or local vLLM, writes Parquet segments via `pymilvus.LocalBulkWriter`, and **uploads each segment to Volume immediately after flush, then deletes the local file** — local disk never exceeds `--segment-size`
4. **Triggers** `bulk_import` to load all uploaded data into the collection

Use `--skip-create` to skip step 1 if the collection already exists.

---

## Pipeline

```
Step 1: Create Collection (skip with --skip-create)
  │
  ▼
Step 2: Ensure Volume exists (Aliyun Beijing)
  │
  ▼
Step 3: Stream JSONL → Embed → Write Parquet → Upload → Delete (per segment)
  │
  │  JSONL (stream line-by-line, ~2 KB RAM per line)
  │    │
  │    ├─ extract caption ──▶ Embedding (API or local vLLM) ──▶ caption_vector
  │    │
  │    ├─ keep all 13 original fields as individual columns
  │    │
  │    ├─ pack entire line ──▶ raw_json (JSON field, ≤64 KB)
  │    │
  │    └─ LocalBulkWriter auto-flushes at --segment-size
  │         │
  │         After each flush_batch(), scan output dir:
  │         ├─ if new .parquet file found → upload to Volume → delete local
  │         └─ at end: writer.commit() → upload remaining → delete local
  │
  │    Single writer instance for the entire run (no recreation).
  │
  ▼
Step 4: bulk_import into Collection (all segments already on Volume)
```

### Segment Lifecycle (upload-as-you-go)

```
  ┌─────────────────┐     ┌────────────────┐     ┌────────────┐
  │ Writer buffer   │     │ Scan output    │     │ Upload &   │
  │ hits chunk_size │────▶│ dir for new    │────▶│ delete     │
  │ → auto-flush    │     │ .parquet files │     │ local file │
  └─────────────────┘     └────────────────┘     └────────────┘
    --segment-size          after flush_batch()      frees disk

At end: writer.commit() flushes remaining buffer → upload → delete

Local disk usage: always ≤ --segment-size (one segment at a time)
```

---

## Input

**JSONL file** — one JSON object per line, with these fields:

| Field                    | Type          |
|--------------------------|---------------|
| `id`                     | string (UUID) |
| `path`                   | string        |
| `height`                 | int           |
| `width`                  | int           |
| `caption`                | string (nested JSON) |
| `caption_version`        | string        |
| `text_ratio`             | float         |
| `craft_bbox_num`         | int           |
| `fused_image`            | float         |
| `fused_image_aesthetic`  | float         |
| `fused_image_technical`  | float         |
| `image_512`              | string (nested JSON) |
| `rand`                   | float         |

---

## Embedding

Two modes selected by `--embed-mode`:

### API mode (`--embed-mode api`, default)

| Item             | Value                                                        |
|------------------|--------------------------------------------------------------|
| Provider         | Any OpenAI-compatible `/v1/embeddings` endpoint              |
| Endpoint         | `--api-base` (default: `https://api.siliconflow.cn/v1/embeddings`) |
| Auth             | Bearer token via `--api-key` or `EMBED_API_KEY` env var      |
| Model            | `--model` (default `Qwen3-Embedding-4B`)                     |
| Input            | `caption` field (full JSON string)                           |
| Output dimension | `--dim` (default 512)                                        |
| Batch size       | `--batch-size` (default 256, API may limit to 32 per request)|
| Rate limit       | retry with exponential backoff (max `--max-retries`)         |
| Concurrency      | `--workers` for parallel API calls (default 8)               |

### Local mode (`--embed-mode local`)

| Item             | Value                                                        |
|------------------|--------------------------------------------------------------|
| Engine           | vLLM offline inference (`pip install vllm`)                  |
| Model            | `--model` (any HuggingFace model, default `Qwen3-Embedding-4B`) |
| Input            | `caption` field (full JSON string)                           |
| Output dimension | `--dim` (default 512)                                        |
| Batch size       | `--batch-size` (default 256, can go higher for GPU throughput) |
| Concurrency      | Managed by vLLM internally; `--workers` is ignored           |
| API key          | Not required                                                 |

---

## Parquet File Schema (15 columns)

Written by `pymilvus.LocalBulkWriter`, one file per auto-flushed segment (at `--segment-size`).
Does **not** contain `autoid` — Zilliz generates it on import.

| Column                   | Parquet Type           | Source                     |
|--------------------------|------------------------|----------------------------|
| `id`                     | UTF-8 string           | original field             |
| `path`                   | UTF-8 string           | original field             |
| `height`                 | Int32                  | original field             |
| `width`                  | Int32                  | original field             |
| `caption`                | UTF-8 string           | original field             |
| `caption_version`        | UTF-8 string           | original field             |
| `text_ratio`             | Float32                | original field             |
| `craft_bbox_num`         | Int32                  | original field             |
| `fused_image`            | Float32                | original field             |
| `fused_image_aesthetic`  | Float32                | original field             |
| `fused_image_technical`  | Float32                | original field             |
| `image_512`              | UTF-8 string           | original field             |
| `rand`                   | Float32                | original field             |
| `raw_json`               | UTF-8 string (JSON)    | entire original JSONL line |
| `caption_vector`         | list\<float32\> × `--dim` | embedding (API or local) |

---

## Zilliz Cloud Collection Schema (16 fields)

| Field                    | DataType             | Constraints                     |
|--------------------------|----------------------|---------------------------------|
| `autoid`                 | Int64                | **primary key**, `auto_id=True` |
| `id`                     | VARCHAR(128)         | —                               |
| `path`                   | VARCHAR(1024)        | —                               |
| `height`                 | Int32                | **scalar index**                |
| `width`                  | Int32                | **scalar index**                |
| `caption`                | VARCHAR(65535)       | —                               |
| `caption_version`        | VARCHAR(32)          | **scalar index**                |
| `text_ratio`             | Float                | —                               |
| `craft_bbox_num`         | Int32                | —                               |
| `fused_image`            | Float                | —                               |
| `fused_image_aesthetic`  | Float                | —                               |
| `fused_image_technical`  | Float                | —                               |
| `image_512`              | VARCHAR(1024)        | —                               |
| `rand`                   | Float                | —                               |
| `raw_json`               | JSON                 | —                               |
| `caption_vector`         | FloatVector(`--dim`) | **vector index (AUTOINDEX)**    |

### Schema–Parquet Mapping

```
Parquet (15 cols)                   Collection (16 fields)
─────────────────                   ──────────────────────
                          ×         autoid        ← Zilliz auto-generates
id                        →        id
path                      →        path
height                    →        height
width                     →        width
caption                   →        caption
caption_version           →        caption_version
text_ratio                →        text_ratio
craft_bbox_num            →        craft_bbox_num
fused_image               →        fused_image
fused_image_aesthetic     →        fused_image_aesthetic
fused_image_technical     →        fused_image_technical
image_512                 →        image_512
rand                      →        rand
raw_json                  →        raw_json
caption_vector            →        caption_vector
```

---

## Collection Creation (Step 1)

The script creates the collection with schema, indexes, and `auto_id=True`. Skip with `--skip-create`.

```python
from pymilvus import MilvusClient, DataType

schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

schema.add_field("autoid",                 DataType.INT64,        is_primary=True)
schema.add_field("id",                     DataType.VARCHAR,      max_length=128)
schema.add_field("path",                   DataType.VARCHAR,      max_length=1024)
schema.add_field("height",                DataType.INT32)
schema.add_field("width",                 DataType.INT32)
schema.add_field("caption",               DataType.VARCHAR,      max_length=65535)
schema.add_field("caption_version",       DataType.VARCHAR,      max_length=32)
schema.add_field("text_ratio",            DataType.FLOAT)
schema.add_field("craft_bbox_num",        DataType.INT32)
schema.add_field("fused_image",           DataType.FLOAT)
schema.add_field("fused_image_aesthetic", DataType.FLOAT)
schema.add_field("fused_image_technical", DataType.FLOAT)
schema.add_field("image_512",             DataType.VARCHAR,      max_length=1024)
schema.add_field("rand",                  DataType.FLOAT)
schema.add_field("raw_json",              DataType.JSON)
schema.add_field("caption_vector",        DataType.FLOAT_VECTOR, dim=1024)

index_params = client.prepare_index_params()
index_params.add_index(field_name="caption_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="height",          index_type="AUTOINDEX")
index_params.add_index(field_name="width",           index_type="AUTOINDEX")
index_params.add_index(field_name="caption_version", index_type="AUTOINDEX")

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params)
```

---

## CLI Configuration

| Param               | CLI flag            | Env var               | Default                                    |
|----------------------|---------------------|-----------------------|--------------------------------------------|
| Input JSONL path     | `--input`           | —                     | required                                   |
| Output directory     | `--output-dir`      | —                     | `./bulk_output`                            |
| Embed mode           | `--embed-mode`      | —                     | `api`                                      |
| Embedding model      | `--model`           | —                     | `Qwen3-Embedding-4B`                      |
| Vector dimension     | `--dim`             | —                     | `512`                                      |
| Batch size           | `--batch-size`      | —                     | `256`                                      |
| API endpoint         | `--api-base`        | —                     | `https://api.siliconflow.cn/v1/embeddings` |
| API key              | `--api-key`         | `EMBED_API_KEY`       | required in api mode only                  |
| Async workers        | `--workers`         | —                     | `8` (api mode only)                        |
| Max retries          | `--max-retries`     | —                     | `3` (api mode only)                        |
| Segment size         | `--segment-size`    | —                     | `128MB`                                    |
| Collection name      | `--collection`      | —                     | required                                   |
| Cluster endpoint     | `--cluster-endpoint`| `ZILLIZ_ENDPOINT`     | required                                   |
| Cluster token        | `--cluster-token`   | `ZILLIZ_TOKEN`        | required                                   |
| Cluster ID           | `--cluster-id`      | `ZILLIZ_CLUSTER_ID`   | required                                   |
| Zilliz API key       | `--zilliz-api-key`  | `ZILLIZ_API_KEY`      | required                                   |
| Volume name          | `--volume-name`     | `ZILLIZ_VOLUME`       | `bulk_import_vol`                          |
| Project ID           | `--project-id`      | `ZILLIZ_PROJECT_ID`   | required                                   |
| Region ID            | `--region-id`       | `ZILLIZ_REGION_ID`    | `ali-cn-beijing`                           |
| Skip create          | `--skip-create`     | —                     | `false`                                    |

Checkpoint is stored automatically at `<output-dir>/checkpoint.json` (no CLI flag).

---

## Memory & Disk Usage

| Resource | Component                       | Usage                     |
|----------|---------------------------------|---------------------------|
| RAM      | JSONL reader                    | ~2 KB (one line)          |
| RAM      | Embedding API text batch        | 32 × ~1.5 KB ≈ 48 KB     |
| RAM      | LocalBulkWriter segment buffer  | up to `--segment-size`    |
| **RAM**  | **Total peak**                  | **≈ `--segment-size`**    |
| Disk     | One Parquet segment (written, pending upload) | up to `--segment-size` |
| Disk     | Checkpoint file                 | < 1 KB                    |
| **Disk** | **Total peak**                  | **≈ `--segment-size`**    |

Both RAM and disk are constant regardless of total file size (500M rows safe).
Segments are uploaded to Volume and deleted locally immediately after auto-flush — no accumulation.
Default `--segment-size` is 128 MB; increase to 512 MB–1 GB for production runs.

---

## Checkpoint & Resume

- Checkpoint saved periodically (every 10,000 lines) to `<output-dir>/checkpoint.json`
- Stores both `line_offset` and `segment_idx` to prevent filename collisions on resume
- On restart, the script auto-detects the checkpoint file and resumes from the saved line offset
- Guarantees no duplicate rows and no re-embedding after crash

---

## Error Handling

| Scenario                          | Behavior                                           |
|-----------------------------------|-----------------------------------------------------|
| Malformed JSONL line              | Log warning, skip line, continue                    |
| Empty / null caption              | Log warning, write zero vector (`[0.0] * dim`)      |
| API 429 (rate limit)              | Retry with exponential backoff, up to `--max-retries` |
| API 5xx / timeout                 | Retry with backoff; abort after `--max-retries`     |
| raw_json exceeds 64 KB            | Log warning, truncate                               |
| Vector contains non-float values  | Force all elements to `float()` before writing      |

---

## Concurrency & Throughput Estimate

| Workers | Texts/sec (est.) | Time for 500M rows |
|---------|-----------------|---------------------|
| 1       | ~100            | ~58 days            |
| 8       | ~800            | ~7 days             |
| 32      | ~3,200          | ~1.8 days           |
| 64      | ~6,400          | ~22 hours           |

---

## Dependencies

```
pymilvus[bulk_writer]>=2.5
tqdm

# Optional (pick one based on --embed-mode):
aiohttp   # required for --embed-mode=api
vllm      # required for --embed-mode=local
```

---

## Non-Goals

- Image downloading or processing
- Deduplication
- Multi-file input (single JSONL → chunked Parquet output)
