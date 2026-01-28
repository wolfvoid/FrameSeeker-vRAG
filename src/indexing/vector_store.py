import chromadb
from chromadb.config import Settings
from FlagEmbedding import FlagModel
import json
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from transformers import logging as transformers_logging

from src.config import Config


class VideoVectorStore:
    def __init__(self):
        transformers_logging.set_verbosity_error()
        # 1. 初始化 ChromaDB (持久化存储)
        print(
            f"[VectorStore] Initializing ChromaDB at {Config.VECTOR_DB_DIR}...")
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)

        # 2. 初始化/获取集合 (Collections)
        self.visual_col = self.client.get_or_create_collection(
            name="visual_frames", metadata={"hnsw:space": "cosine"})
        self.audio_col = self.client.get_or_create_collection(
            name="audio_transcripts", metadata={"hnsw:space": "cosine"})

        # 3. 加载文本 Embedding 模型 (BGE-M3) 用于计算字幕向量
        # 注意：视觉向量在预处理阶段已经算好存在 JSON 里了，但字幕的还没算
        print(
            f"[VectorStore] Loading BGE-M3 model from {Config.MODEL_PATH['embedding']}...")
        self.text_encoder = FlagModel(
            Config.MODEL_PATH['embedding'],
            query_instruction_for_retrieval="为文本生成向量以用于检索: ",
            use_fp16=True,
            devices=["cuda:0"]
        )

    def ingest_video(self, json_path: str):
        """将单个视频的 JSON 数据入库"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        video_id = data['video_id']
        print(f"[VectorStore] Ingesting {video_id}...")

        # === 1. 处理视觉帧 (Visual Frames) ===
        # JSON里已经有 embedding 了，直接存
        frames = data['frames']
        if frames:
            v_ids = []
            v_embeddings = []
            v_metadatas = []

            for frame in frames:
                # ID: video_id + timestamp
                fid = f"{video_id}_v_{frame['timestamp']}"
                v_ids.append(fid)
                v_embeddings.append(frame['embedding'])
                v_metadatas.append({
                    "video_id": video_id,
                    "timestamp": frame['timestamp'],
                    "frame_path": frame['frame_path'],
                    "type": "visual"
                })

            # 批量写入 (Batch upsert)
            batch_size = 500
            for i in range(0, len(v_ids), batch_size):
                self.visual_col.upsert(
                    ids=v_ids[i: i+batch_size],
                    embeddings=v_embeddings[i: i+batch_size],
                    metadatas=v_metadatas[i: i+batch_size]
                )

        # === 2. 处理字幕 (Audio Transcripts) ===
        # JSON里只有 text，需要现场算 embedding
        transcripts = data['transcript']
        if transcripts:
            a_ids = []
            a_texts = []     # 存原文
            a_metadatas = []

            for i, sub in enumerate(transcripts):
                aid = f"{video_id}_a_{i}"
                a_ids.append(aid)
                a_texts.append(sub['text'])
                a_metadatas.append({
                    "video_id": video_id,
                    "start": sub['start'],
                    "end": sub['end'],
                    "text": sub['text'],  # 把原文也存进 metadata，方便检索后直接拿
                    "type": "audio"
                })

            # 计算 Embedding
            print(
                f"  - Computing embeddings for {len(a_texts)} transcript segments...")
            a_embeddings = self.text_encoder.encode(a_texts)

            # 批量写入
            for i in range(0, len(a_ids), batch_size):
                self.audio_col.upsert(
                    ids=a_ids[i: i+batch_size],
                    # numpy 转 list
                    embeddings=a_embeddings[i: i+batch_size].tolist(),
                    metadatas=a_metadatas[i: i+batch_size]
                )

        print(
            f"[VectorStore] Finished {video_id}: {len(frames)} frames, {len(transcripts)} subs.")


if __name__ == "__main__":
    store = VideoVectorStore()
    processed_dir = Config.PROCESSED_DIR
    json_files = [os.path.join(processed_dir, f) for f in os.listdir(
        processed_dir) if f.endswith('.json')]

    for jf in tqdm(json_files, desc="Ingesting videos"):
        store.ingest_video(jf)
