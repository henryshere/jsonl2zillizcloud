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

## Configuration Architecture

The script uses a `PipelineConfig` dataclass to centralize all pipeline settings. It is built once from CLI args at startup and passed to every function that needs configuration. This design avoids global variables, keeps functions testable and self-contained, and supports future modularization into separate files (e.g. `schema.py`, `embedder.py`, `writer.py`).

### PipelineConfig fields

| Field                  | Type   | CLI source                    | Default | Purpose                                  |
|------------------------|--------|-------------------------------|---------|------------------------------------------|
| `dim`                  | int    | `--dim`                       | `512`   | Embedding vector dimension               |
| `include_raw_json`     | bool   | inverse of `--no-raw-json`    | `True`  | Whether to include `raw_json` field      |
| `include_caption_str`  | bool   | inverse of `--no-caption-str` | `True`  | Whether to include `caption_str` field   |
| `include_caption_json` | bool   | inverse of `--no-caption-json`| `True`  | Whether to include `caption_json` field (and run JSON parsing) |

New pipeline-level flags are added as fields here — no function signatures need to change.

### How it flows

```
CLI args → PipelineConfig (built once in run())
                │
                ├── build_schema(config)        → schema with/without raw_json
                ├── create_collection(config)   → uses build_schema
                ├── make_row(config)            → includes/excludes raw_json
                └── stream_embed_write(config)  → passes to build_schema + make_row
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

## Parquet File Schema (up to 16 columns — `caption_str`, `caption_json`, `raw_json` individually excludable)

Written by `pymilvus.LocalBulkWriter`, one file per auto-flushed segment (at `--segment-size`).
Does **not** contain `autoid` — Zilliz generates it on import.

| Column                   | Parquet Type           | Source                     |
|--------------------------|------------------------|----------------------------|
| `id`                     | UTF-8 string           | original field             |
| `path`                   | UTF-8 string           | original field             |
| `height`                 | Int32                  | original field             |
| `width`                  | Int32                  | original field             |
| `caption_str`            | UTF-8 string           | original `caption` field (excluded with `--no-caption-str`) |
| `caption_json`           | UTF-8 string (JSON)    | parsed caption dict (excluded with `--no-caption-json`) |
| `caption_version`        | UTF-8 string           | original field             |
| `text_ratio`             | Float32                | original field             |
| `craft_bbox_num`         | Int32                  | original field             |
| `fused_image`            | Float32                | original field             |
| `fused_image_aesthetic`  | Float32                | original field             |
| `fused_image_technical`  | Float32                | original field             |
| `image_512`              | UTF-8 string           | original field             |
| `rand`                   | Float32                | original field             |
| `raw_json`               | UTF-8 string (JSON)    | entire original JSONL line (excluded with `--no-raw-json`) |
| `caption_vector`         | list\<float32\> × `--dim` | embedding (API or local) |

---

## Zilliz Cloud Collection Schema (up to 17 fields — `caption_str`, `caption_json`, `raw_json` individually excludable)

| Field                    | DataType             | Constraints                     |
|--------------------------|----------------------|---------------------------------|
| `autoid`                 | Int64                | **primary key**, `auto_id=True` |
| `id`                     | VARCHAR(128)         | —                               |
| `path`                   | VARCHAR(1024)        | —                               |
| `height`                 | Int32                | **scalar index**                |
| `width`                  | Int32                | **scalar index**                |
| `caption_str`            | VARCHAR(65535)       | excluded with `--no-caption-str` |
| `caption_json`           | JSON                 | excluded with `--no-caption-json` |
| `caption_version`        | VARCHAR(32)          | **scalar index**                |
| `text_ratio`             | Float                | —                               |
| `craft_bbox_num`         | Int32                | —                               |
| `fused_image`            | Float                | —                               |
| `fused_image_aesthetic`  | Float                | —                               |
| `fused_image_technical`  | Float                | —                               |
| `image_512`              | VARCHAR(1024)        | —                               |
| `rand`                   | Float                | —                               |
| `raw_json`               | JSON                 | excluded with `--no-raw-json`   |
| `caption_vector`         | FloatVector(`--dim`) | **vector index (AUTOINDEX)**    |

> **Breaking change (vs pre-rename):** The raw-string caption field is now named
> `caption_str` (previously `caption`). Existing collections created before this
> rename cannot receive new data from the renamed schema — `bulk_import` will fail
> due to field-name mismatch. Either drop and recreate the collection, or keep
> using the old schema version for existing data.

### Schema–Parquet Mapping

```
Parquet (16 cols)                   Collection (17 fields)
─────────────────                   ──────────────────────
                          ×         autoid        ← Zilliz auto-generates
id                        →        id
path                      →        path
height                    →        height
width                     →        width
caption_str               →        caption_str
caption_json              →        caption_json
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
| Exclude caption_str  | `--no-caption-str`  | —                     | `false`                                    |
| Exclude caption_json | `--no-caption-json` | —                     | `false`                                    |
| Exclude raw_json     | `--no-raw-json`     | —                     | `false`                                    |

Checkpoint is stored automatically at `<output-dir>/checkpoint.json` (no CLI flag).

The three `--no-*` field flags (`--no-caption-str`, `--no-caption-json`, `--no-raw-json`) independently exclude their respective fields from both the collection schema and parquet output. `--no-caption-json` additionally skips per-row JSON parsing, sanitization, and size-check. All three flags must be consistent across runs targeting the same collection — mismatched schemas will cause `bulk_import` to fail.

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

- Multi-file input (single JSONL → chunked Parquet output)
- **Variable / inferred JSONL schemas** — see discussion below

### Variable / inferred JSONL schemas (deliberately out of scope)

Today the pipeline has a **hardcoded schema** baked into `config.py`: 16 named fields
with fixed types, an explicit `make_row()` mapping in `writer.py`, and flag-gated
inclusion for `caption_str` / `caption_json` / `raw_json`. Extra JSONL keys are silently
dropped; missing keys fall back to hardcoded defaults. Adapting the tool to a new
dataset shape requires editing Python in three places (`config.py`, `writer.py`,
`__main__.py`).

This is a deliberate trade-off. Making the schema configurable is feasible but
non-trivial, and the current tight coupling buys us type safety, scalar indexes on
hot fields, per-row CPU efficiency (no schema interpretation in the hot loop),
constant-memory streaming over 500M rows, and fail-fast behavior on shape
mismatches. A generic schema system must preserve all five of these.

#### What we'd need to preserve

- Typed Milvus fields — not everything stuffed into a single JSON blob
- Scalar indexes on queryable fields (`height`, `caption_version`, …)
- Zero per-row schema interpretation — branches resolved at startup, not per row
- Constant memory — no full-file scans at startup
- Fail-fast on shape mismatches — catch problems at row 1, not at row 300M
- Reproducibility — two runs of the same input must produce identical schemas

#### Options considered

| Option | Summary | Verdict |
|---|---|---|
| **Milvus dynamic field** (`enable_dynamic_field=True`) | Declare only primary key + vector + embed source; everything else goes to `$meta`. | Primary mechanism — kills typed indexes. Useful as escape hatch. |
| **Schema inference from sample** | Read first N lines, infer types and max lengths, build schema at startup. | Non-reproducible; max_length wrong; silent type drift. Useful only as a helper that emits a starter descriptor. |
| **Declarative schema descriptor** (YAML/JSON) | User commits a `schema.yaml` describing fields, types, sources, transforms, index hints, and embedding source. Parsed once at startup. | Recommended primary mechanism. |
| **Hybrid: inference helper + descriptor** | `--infer-schema` subcommand samples the JSONL and emits a starter descriptor; user edits it; main pipeline consumes it. | Recommended. Combines zero-config discovery with determinism. |

#### Sketch of the recommended approach

```yaml
# schema.yaml — committed alongside the dataset
collection:
  name: my_collection
  auto_id: true
  primary_key: autoid

fields:
  - name: id
    type: VARCHAR
    max_length: 128
  - name: height
    type: INT32
    index: AUTOINDEX
  - name: caption_str
    source: caption
    type: VARCHAR
    max_length: 65535
    include: true
  - name: caption_json
    source: caption
    type: JSON
    transform: parse_json
    transform_args:
      sanitize_keys: true
      max_bytes: 65536
  - name: raw_json
    source: $record          # sentinel: the whole JSONL line
    type: JSON
    transform: truncate_utf8
    transform_args:
      max_bytes: 65536

vector:
  name: caption_vector
  source: caption
  dim: 512
  index: AUTOINDEX
  metric: COSINE
```

**Transforms are named references** to built-in functions (`parse_json`, `truncate_utf8`,
`coerce_int`, `coerce_float`, `identity`) — users never write Python callbacks. Adding
a transform means adding a function to a registry module. No `eval`, no `exec`, no
user-supplied code paths.

**The descriptor is compiled into a closure** at startup so per-row overhead is
identical to today's hand-written `make_row`. Branches for excluded fields simply
don't exist in the generated code.

**A fail-fast validator** runs against the first ~100 rows of the JSONL before any
parquet is written: every `source` field exists, declared types match observed types,
`VARCHAR max_length` is ≥ observed max, `vector.source` is non-empty in at least one
row. Catches shape mismatches in ≤ 1 second.

#### Migration from today's hardcoded schema

| Current | After |
|---|---|
| `config.py: build_schema(config)` hardcoded | `build_schema(descriptor)` |
| `writer.py: make_row()` hardcoded map | Generated row-builder closure |
| `PipelineConfig.include_caption_*` | `include: true/false` per field in descriptor |
| `--no-caption-str` / `--no-caption-json` / `--no-raw-json` CLI flags | Edit descriptor |
| `--dim` CLI flag | `vector.dim` in descriptor |
| Python source edits for each new dataset | Edit `schema.yaml` |

Today's behavior becomes the default descriptor shipped with the tool
(`schemas/default.yaml`) — existing users see no behavior change.

#### Phased roadmap (if and when we decide to build this)

1. **Descriptor parser + `build_schema(descriptor)`** — small; replaces current `build_schema`.
2. **Compiled row-builder + transform registry** — medium; replaces `make_row`.
3. **`--infer-schema` subcommand** — small; samples + emits starter YAML.
4. **Fail-fast validator** — small; single-pass over a sample of the JSONL.
5. **Dynamic-field escape hatch** — tiny; `enable_dynamic_field: true` in descriptor.

Phase 1 + 2 alone delivers 80% of the value; phases are independently shippable.

#### What to avoid

- Infer at every run — non-reproducible, slow, silently drifts
- Milvus dynamic field as the primary mechanism — kills typed indexes
- User-supplied Python (`lambda`, `exec`, callable paths) in descriptors
- Skipping the validator — fail-fast is the main win over today's approach
- Auto-magic — explicit beats clever for batch pipelines

#### Open questions (for whoever picks this up)

1. YAML vs JSON vs TOML for the descriptor format?
2. Schema descriptor location — alongside the JSONL, alongside the script, or CLI flag?
3. Multiple vector fields in one collection? (e.g. image + text embeddings)
4. Nested field flattening — `record["meta"]["source"]` → column `meta_source`, or pre-flatten?
5. Descriptor versioning — embed `version: 1` so the format can evolve safely?
