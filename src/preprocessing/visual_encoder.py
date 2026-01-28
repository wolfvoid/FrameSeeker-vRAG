from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from typing import List
import numpy as np


class VisualEncoder:
    def __init__(self, model_path: str, device: str):
        print(f"[VisualEncoder] Loading SigLIP model on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """批量提取图片特征"""
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # SigLIP 预处理
        inputs = self.processor(
            images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # 获取 image features
            outputs = self.model.get_image_features(**inputs)
            # 归一化 (对于 Cosine Similarity 很重要)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        return outputs.cpu().numpy()  # 转回 CPU 存 numpy
