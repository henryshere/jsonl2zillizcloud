export EMBED_API_KEY=sk-tyhuqxjkkjhuslsfsrxczhwwrkavffjbyhhdsimpnfggunjj
export ZILLIZ_API_KEY=0dbcde6c566da61db826def9c9d406edae946e6bdc98508171548a3813a94d135c4ab2e82b3e6b003e0d7d0510fa456cb44d1dd9
export ZILLIZ_TOKEN=0dbcde6c566da61db826def9c9d406edae946e6bdc98508171548a3813a94d135c4ab2e82b3e6b003e0d7d0510fa456cb44d1dd9
export ZILLIZ_ENDPOINT=https://in01-9e68d1ad8e364a2.ali-cn-beijing.vectordb.zilliz.com.cn:19530
export ZILLIZ_CLUSTER_ID=in01-9e68d1ad8e364a2
export ZILLIZ_PROJECT_ID=proj-7987f39e0bfa02ea9bf116

python3 ./jsonl_to_parquet.py \
    --input            ./1.jsonl \
    --output-dir       ./bulk_output \
    --collection       test4vidu \
    --embed-mode       local \
    --model            ~/qwen_emb/model/qwen/Qwen3-Embedding-4B \
    --dim              512 \
    --batch-size       256

