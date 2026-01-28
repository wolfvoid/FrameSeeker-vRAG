import os
import base64
from io import BytesIO
from typing import List, Dict
from PIL import Image
from openai import OpenAI

class VLLMGenerator:
    def __init__(self, api_url="http://localhost:8000/v1", api_key="EMPTY"):
        print(
            f"[Generator] Initializing OpenAI Client connecting to {api_url}...")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
        self.model_name = "qwen-vl"  # 对应启动命令里的 --served-model-name

    def _encode_image_to_base64(self, img_path: str) -> str:
        """辅助函数：将图片文件转为 Base64 字符串"""
        try:
            img = Image.open(img_path).convert("RGB")
            # 缩放图片以加快传输 (可选，大图传输会慢)
            # img.thumbnail((1024, 1024))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"[Error] Encoding image failed: {e}")
            return None

    def format_messages(self, query: str, retrieval_results: List[Dict]):
        """构建 OpenAI 格式的消息"""

        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful video assistant. "
                "Answer the user's question based on the provided visual frames and audio transcripts. "
                "CRITICAL: You must cite the timestamp for every claim you make, e.g., '[10.5s]'. "
                "If the answer is not in the context, say you don't know."
            )
        }

        user_content = []
        user_content.append(
            {"type": "text", "text": "Here is the retrieved context from the video:\n\n"})

        # 1. 注入视觉信息
        user_content.append(
            {"type": "text", "text": "--- Visual Frames ---\n"})
        visual_results = [
            r for r in retrieval_results if r['type'] == 'visual']

        for res in visual_results:
            base64_img = self._encode_image_to_base64(res['path'])
            if base64_img:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_img}
                })
                user_content.append({
                    "type": "text",
                    "text": f"\n(Image at timestamp: {res['timestamp']}s)\n"
                })

        # 2. 注入听觉信息
        user_content.append(
            {"type": "text", "text": "\n--- Audio Transcripts ---\n"})
        audio_results = [r for r in retrieval_results if r['type'] == 'audio']

        for res in audio_results:
            user_content.append({
                "type": "text",
                "text": f"[{res['start']}s - {res['end']}s]: {res['content']}\n"
            })

        # 3. 注入问题
        user_content.append(
            {"type": "text", "text": f"\n\nUser Question: {query}"})

        return [system_msg, {"role": "user", "content": user_content}]

    def generate(self, query: str, retrieval_results: List[Dict]) -> str:
        """发送 API 请求"""
        messages = self.format_messages(query, retrieval_results)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=512,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Generator] API Error: {e}")
            return f"Error communicating with model: {e}"

# 测试脚本
if __name__ == "__main__":
    generator = VLLMGenerator()

    # 模拟检索结果
    mock_results = [
        {
            "type": "audio", "start": 10.0, "end": 15.0,
            "content": "AlexNet introduced ReLU activation function to solve gradient vanishing."
        },
        {
            "type": "visual", "timestamp": 12.0, "path": "/home/meibingyin/Projects/vrag/FrameSeeker-vRAG/data/processed_frames/9年后重读深度学习奠基作之一：AlexNet【论文精读·2】 - 1.9年后重读深度学习奠基作之一：AlexNet【论文精读·2】(Av208532381,P1)_2s.jpg"
        }
    ]

    query = "Why did AlexNet use ReLU?"
    print(f"\n[Test] Generating answer for: {query}")

    answer = generator.generate(query, mock_results)
    print("-" * 50)
    print(answer)
    print("-" * 50)
