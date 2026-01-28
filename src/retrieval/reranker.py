import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import Config


class BGEReranker:
    def __init__(self, device: str = "cuda:0"):
        print(
            f"[Reranker] Loading BGE-Reranker from {Config.MODEL_PATH['rerank']} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_PATH['rerank'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_PATH['rerank']
        ).to(device).eval()

        # 开启半精度加速
        if torch.cuda.is_available():
            self.model.half()

    def compute_score(self, query: str, texts: list[str]) -> list[float]:
        """
        计算 Query 和一堆 Text 的相关性分数
        返回: list[float] 分数越高越相关
        """
        if not texts:
            return []

        pairs = [[query, text] for text in texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            # 推理
            scores = self.model(
                **inputs, return_dict=True).logits.view(-1,).float()

        # 将分数归一化到 0-1 之间 (Sigmoid) 方便和向量分数融合
        scores = torch.sigmoid(scores)
        return scores.cpu().numpy().tolist()
