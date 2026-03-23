# PRD: JSONL to Zilliz Cloud with Embedding Enrichment

## Overview

A Python CLI script that:

1. **Creates** the Zilliz Cloud collection (with schema, indexes, and `auto_id=True`)
2. **Streams** a massive JSONL file (up to 500M rows) in constant memory
3. **Embeds** the `caption` field by calling the SiliconFlow REST API
4. **Writes** chunked Parquet files (with pre-computed vectors) via `pymilvus.LocalBulkWriter`
5. **Uploads** Parquet chunks to a Zilliz Cloud Volume (Alibaba Cloud, Beijing region)
6. **Triggers** `bulk_import` to load data into the collection

Use `--skip-create` to skip step 1 if the collection already exists.

---

## Pipeline

```
Step 1: Create Collection (skip with --skip-create)
  │
  ▼
Step 2: Stream JSONL → Embed → Write Parquet
  │
  │  JSONL (stream line-by-line, ~2 KB RAM per line)
  │    │
  │    ├─ extract caption ──▶ SiliconFlow API (batch=32) ──▶ caption_vector
  │    │
  │    ├─ keep all 13 original fields as individual columns
  │    │
  │    ├─ pack entire line ──▶ raw_json (JSON field, ≤64 KB)
  │    │
  │    └─ LocalBulkWriter (1 GB segment) ──▶ part-*.parquet
  │
  ▼
Step 3: Upload to Volume (Aliyun Beijing)
  │
  ▼
Step 4: bulk_import into Collection
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

| Item             | Value                                            |
|------------------|--------------------------------------------------|
| Provider         | SiliconFlow REST API                             |
| Endpoint         | `POST https://api.siliconflow.cn/v1/embeddings`  |
| Auth             | Bearer token via `SILICONFLOW_API_KEY` env var   |
| Model            | configurable, default `BAAI/bge-m3`              |
| Input            | `caption` field (full JSON string)               |
| Output dimension | 1024 (for bge-m3)                                |
| Batch size       | up to 32 texts per request (API limit)           |
| Rate limit       | retry with exponential backoff (max 3 retries)   |
| Concurrency      | `--workers` flag for parallel API calls (default 8) |

---

## Parquet File Schema (15 columns)

Written by `pymilvus.LocalBulkWriter`, one file per 1 GB segment.
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
| `caption_vector`         | list\<float32\> × 1024 | SiliconFlow embedding      |

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
| `caption_vector`         | FloatVector(1024)    | **vector index (AUTOINDEX)**    |

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

| Param               | CLI flag           | Env var               | Default                        |
|----------------------|--------------------|-----------------------|--------------------------------|
| Input JSONL path     | `--input`          | —                     | required                       |
| Output directory     | `--output-dir`     | —                     | `./bulk_output`                |
| SiliconFlow API key  | —                  | `SILICONFLOW_API_KEY` | required                       |
| Embedding model      | `--model`          | —                     | `BAAI/bge-m3`                  |
| Vector dimension     | `--dim`            | —                     | `1024`                         |
| API batch size       | `--batch-size`     | —                     | `32`                           |
| Async workers        | `--workers`        | —                     | `8`                            |
| Segment size         | `--segment-size`   | —                     | `1GB`                          |
| Max retries          | `--max-retries`    | —                     | `3`                            |
| Checkpoint file      | `--checkpoint`     | —                     | `<output-dir>/checkpoint.json` |
| Collection name      | `--collection`     | `ZILLIZ_COLLECTION`   | required                       |
| Cluster endpoint     | `--cluster-endpoint`| `ZILLIZ_ENDPOINT`    | required                       |
| Cluster token        | `--cluster-token`  | `ZILLIZ_TOKEN`        | required                       |
| Cluster ID           | `--cluster-id`     | `ZILLIZ_CLUSTER_ID`   | required                       |
| Zilliz API key       | `--zilliz-api-key` | `ZILLIZ_API_KEY`      | required                       |
| Volume name          | `--volume-name`    | `ZILLIZ_VOLUME`       | `bulk_import_vol`              |
| Project ID           | `--project-id`     | `ZILLIZ_PROJECT_ID`   | required                       |
| Region ID            | `--region-id`      | `ZILLIZ_REGION_ID`    | `ali-cn-beijing`               |
| Skip create          | `--skip-create`    | —                     | `false`                        |

---

## Memory Usage

| Component                       | Memory               |
|---------------------------------|----------------------|
| JSONL reader                    | ~2 KB (one line)     |
| SiliconFlow API text batch      | 32 × ~1.5 KB ≈ 48 KB|
| LocalBulkWriter segment buffer  | up to 1 GB           |
| **Total peak**                  | **~1 GB**            |

Constant regardless of total file size (500M rows safe).

---

## Checkpoint & Resume

- After each Parquet segment is flushed, write the current JSONL line offset to `checkpoint.json`
- On restart with `--checkpoint`, skip to the saved offset and resume
- Guarantees no duplicate rows and no re-embedding after crash

---

## Error Handling

| Scenario                          | Behavior                                           |
|-----------------------------------|-----------------------------------------------------|
| Malformed JSONL line              | Log warning, skip line, continue                    |
| Empty / null caption              | Log warning, write null vector                      |
| SiliconFlow 429 (rate limit)      | Retry with exponential backoff, up to 3 times       |
| SiliconFlow 5xx / timeout         | Retry with backoff; abort after max retries         |
| raw_json exceeds 64 KB            | Log warning, truncate or skip                       |
| Upload failure                    | Retry 3× with backoff; abort on persistent failure  |

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
pymilvus>=2.5
aiohttp
tqdm
tenacity
```

---

## Non-Goals

- Image downloading or processing
- Deduplication
- Multi-file input (single JSONL → chunked Parquet output)
