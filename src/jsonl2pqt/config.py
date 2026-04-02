"""Configuration constants, PipelineConfig dataclass, and schema builder."""

from __future__ import annotations

from dataclasses import dataclass

from pymilvus import MilvusClient, DataType

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
