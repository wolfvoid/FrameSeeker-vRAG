from modelscope.hub.snapshot_download import snapshot_download

model_list =[
    "google/siglip2-base-patch16-224",
    "openai-mirror/whisper-large-v3-turbo",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-VL-8B-Instruct"
]

for model_id in model_list:
    local_dir = '/data3/meibingyin/models'
    path = snapshot_download(model_id, cache_dir=local_dir)
    print(f"模型 {model_id} 已下载至: {path}")
