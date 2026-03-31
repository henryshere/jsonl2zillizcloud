from pymilvus import MilvusClient

ZILLIZ_TOKEN = "0dbcde6c566da61db826def9c9d406edae946e6bdc98508171548a3813a94d135c4ab2e82b3e6b003e0d7d0510fa456cb44d1dd9"
ZILLIZ_ENDPOINT = "https://in01-9e68d1ad8e364a2.ali-cn-beijing.vectordb.zilliz.com.cn:19530"

client = MilvusClient(
    uri=ZILLIZ_ENDPOINT,
    token=ZILLIZ_TOKEN
)

filter_expr = """
(width >= 1280 or height >= 1280) 
and text_ratio <= 0.05
and fused_image >= 50 
"""
# and JSONExtractString(caption, 'Additional Information') not in [
#     'Subtitle', 
#     'Danmaku', 
#     'Comment (Live Chat)', 
#     'QR Code', 
#     'Closing Credit'
# ]

def get_count(client, collection_name, expr):
    res = client.query(
        collection_name=collection_name,
        filter=expr,
        output_fields=["count(*)"]
    )
    return res[0]["count(*)"] if res else 0

count1 = get_count(client, "test4vidu", filter_expr)
# count2 = get_count(client, "caption_rollback_image_part1", filter_expr)
# total = count1 + count2

print(f"part1 : {count1}")
# print(f"part2 : {count2}")
# print(f"UNION ALL : {total}")
