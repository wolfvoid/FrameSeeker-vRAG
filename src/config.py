import os
import torch

class Config:
    # ================= 路径配置 =================
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 原始数据路径
    VIDEO_DIR = os.path.join(PROJECT_ROOT, "data/raw_videos")
    # 预处理结果保存路径 (元数据 JSON)
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed_metadata")
    # 关键帧图片保存路径
    FRAMES_DIR = os.path.join(PROJECT_ROOT, "data/processed_frames")
    # 向量库持久化路径
    VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data/vector_db")

    # 模型加载路径
    MODEL_PATH = {
        "visual": "/data3/meibingyin/models/google/siglip2-base-patch16-224",
        "audio": "/data3/meibingyin/models/openai-mirror/whisper-large-v3-turbo",
        "embedding": "/data3/meibingyin/models/BAAI/bge-m3",
        "rerank": "/data3/meibingyin/models/BAAI/bge-reranker-v2-m3",
        "llm": "/data3/meibingyin/models/Qwen/Qwen3-VL-8B-Instruct"
    }

    # ================= 显卡分配策略 (核心) =================
    # 预处理阶段：使用 0-7 号卡并行处理 (每张卡独立跑 Whisper+SigLIP)
    PREPROCESS_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

    # 在线推理阶段：使用 8,9 号卡做 vLLM 张量并行
    INFERENCE_GPUS = [8, 9]

    # ================= 参数配置 =================
    FRAME_INTERVAL = 1  # 每几秒抽一帧 (秒)

    # 自动创建目录
    for d in [PROCESSED_DIR, FRAMES_DIR, VECTOR_DB_DIR]:
        os.makedirs(d, exist_ok=True)
