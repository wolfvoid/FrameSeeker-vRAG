import chromadb
from FlagEmbedding import FlagModel
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch
import numpy as np
from typing import List, Dict

from src.config import Config
from src.retrieval.reranker import BGEReranker


class VideoSearchEngine:
    def __init__(self, device: str = "cuda:0"):
        print("[SearchEngine] Initializing...")
        self.device = device  # 推理用显卡

        # 1. 连接数据库
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
        self.visual_col = self.client.get_collection("visual_frames")
        self.audio_col = self.client.get_collection("audio_transcripts")

        # 2. 加载 BGE-M3 (用于听觉/字幕检索)
        print(f"[SearchEngine] Loading BGE-M3 for Audio Search...")
        self.audio_query_encoder = FlagModel(
            Config.MODEL_PATH['embedding'],
            query_instruction_for_retrieval="为文本生成向量以用于检索: ",
            use_fp16=True,
            device=self.device
        )

        # 3. 加载 SigLIP (用于视觉检索 - 核心修正点)
        # 需要 SigLIP 的文本编码器把 Query 变成视觉空间的向量
        print(f"[SearchEngine] Loading SigLIP for Visual Search...")
        self.siglip_model = AutoModel.from_pretrained(
            Config.MODEL_PATH['visual']).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(
            Config.MODEL_PATH['visual'])

        # 4. 加载重排器
        self.reranker = BGEReranker(device=self.device)

    def get_visual_query_embedding(self, query: str) -> List[float]:
        """使用 SigLIP 生成视觉兼容的 Query 向量"""
        # SigLIP 的文本处理
        inputs = self.siglip_processor(
            text=[query], return_tensors="pt", padding="max_length", max_length=64).to(self.device)
        with torch.no_grad():
            # 获取文本特征
            text_features = self.siglip_model.get_text_features(**inputs)
            # 归一化 (SigLIP 训练时用了归一化，这里必须对应)
            text_features = text_features / \
                text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

    def search(self, query: str, top_k: int = 5):
        print(f"\n[Search] Query: {query}")

        # === 1. 生成双路 Query 向量 ===
        # 路 A: 视觉向量 (SigLIP Space, ~1152 dim)
        visual_q_vec = self.get_visual_query_embedding(query)
        # 路 B: 听觉向量 (BGE Space, 1024 dim)
        audio_q_vec = self.audio_query_encoder.encode(query).tolist()

        # === 2. 视觉召回 (Visual Recall) ===
        try:
            v_results = self.visual_col.query(
                query_embeddings=[visual_q_vec],  # 用 SigLIP 向量搜 SigLIP 库
                n_results=10,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            print(f"[Error] Visual search failed: {e}")
            v_results = {'ids': [[]]}

        visual_candidates = []
        if v_results['ids'] and v_results['ids'][0]:
            for i, vid in enumerate(v_results['ids'][0]):
                meta = v_results['metadatas'][0][i]
                dist = v_results['distances'][0][i]
                score = 1 - dist

                visual_candidates.append({
                    "type": "visual",
                    "score": score,
                    "video_id": meta['video_id'],
                    "timestamp": meta['timestamp'],
                    "path": meta['frame_path'],
                    "content": "[Visual Frame]",
                    "display_info": f"Frame at {meta['timestamp']}s"
                })

        # === 3. 听觉召回 (Audio Recall) ===
        try:
            a_results = self.audio_col.query(
                query_embeddings=[audio_q_vec],  # 用 BGE 向量搜 BGE 库
                n_results=10,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            print(f"[Error] Audio search failed: {e}")
            a_results = {'ids': [[]]}

        audio_candidates = []
        texts_to_rerank = []

        if a_results['ids'] and a_results['ids'][0]:
            for i, aid in enumerate(a_results['ids'][0]):
                meta = a_results['metadatas'][0][i]
                text = meta['text']

                candidate = {
                    "type": "audio",
                    "video_id": meta['video_id'],
                    "start": meta['start'],
                    "end": meta['end'],
                    "content": text,
                    "display_info": f"Sub: {text} ({meta['start']}-{meta['end']}s)"
                }
                audio_candidates.append(candidate)
                texts_to_rerank.append(text)

        # === 4. 听觉重排 (Rerank) ===
        if texts_to_rerank:
            rerank_scores = self.reranker.compute_score(query, texts_to_rerank)
            for i, score in enumerate(rerank_scores):
                # 融合策略：BGE向量分 + 重排分。这里简单直接用重排分替代，因为它更准
                audio_candidates[i]['score'] = score

        # === 5. 混合排序 ===
        # all_candidates = visual_candidates + audio_candidates
        # all_candidates.sort(key=lambda x: x['score'], reverse=True)

        # return all_candidates[:top_k]
        # 1. 视觉结果按分数排序
        visual_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_visual = visual_candidates[:3] # 强制取前3张图

        # 2. 听觉结果按分数排序
        audio_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_audio = audio_candidates[:3]   # 强制取前3段话

        # 3. 结果展示逻辑
        # 拼在一起返回，但是保留类型标签，方便前端展示
        final_results = top_visual + top_audio

        # 打印调试信息 (看看视觉分数到底是多少)
        print("\n--- Debug Info ---")
        if visual_candidates:
            print(f"Top Visual Score: {visual_candidates[0]['score']:.4f}")
        else:
            print("No visual candidates found.")

        return final_results


if __name__ == "__main__":
    engine = VideoSearchEngine()
    test_query = "文中讲了哪些重要的图"
    results = engine.search(test_query, top_k=5)

    print("-" * 50)
    for i, res in enumerate(results):
        print(
            f"Rank {i+1} [{res['type']} | Score: {res['score']:.4f}]: {res['display_info']}")
