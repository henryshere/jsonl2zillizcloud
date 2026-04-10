#!/bin/bash
PROJECT_DIR="/Users/zilliz/Downloads/playground/cksrpt"
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
cd "$PROJECT_DIR"

export EMBED_API_KEY=sk-tyhuqxjkkjhuslsfsrxczhwwrkavffjbyhhdsimpnfggunjj
export ZILLIZ_API_KEY=0dbcde6c566da61db826def9c9d406edae946e6bdc98508171548a3813a94d135c4ab2e82b3e6b003e0d7d0510fa456cb44d1dd9
export ZILLIZ_TOKEN=0dbcde6c566da61db826def9c9d406edae946e6bdc98508171548a3813a94d135c4ab2e82b3e6b003e0d7d0510fa456cb44d1dd9
export ZILLIZ_ENDPOINT=https://in01-9e68d1ad8e364a2.ali-cn-beijing.vectordb.zilliz.com.cn:19530
export ZILLIZ_CLUSTER_ID=in01-9e68d1ad8e364a2
export ZILLIZ_PROJECT_ID=proj-7987f39e0bfa02ea9bf116

python3 -m jsonl2pqt \
    --input            /Users/zilliz/Downloads/playground/vidu/1.jsonl \
    --output-dir       /Users/zilliz/Downloads/playground/vidu/bulk_output \
    --collection       test4vidu \
    --embed-mode       api \
    --model            BAAI/bge-m3 \
    --api-base         https://api.siliconflow.cn/v1/embeddings \
    --batch-size       8 \
    --dim              1024 \
    --workers          2 \
    --no-raw-json

