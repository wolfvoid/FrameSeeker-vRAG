import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Dict
from tqdm import tqdm

class ASRWorker:
    def __init__(self, model_path: str, device: str):
        print(f"[ASRWorker] Loading Whisper model on {device}...")
        # 1. 加载模型 (半精度 FP16 以加速)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        # 2. 加载处理器
        processor = AutoProcessor.from_pretrained(model_path)

        # 3. 创建流水线
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            dtype=self.torch_dtype,
            device=device,
            ignore_warning=True,
        )

    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        执行转录并适配输出格式
        """
        # language="chinese" 强制指定中文，防止识别成别的
        # generate_kwargs 可以控制生成细节
        try:
            result = self.pipe(audio_path, generate_kwargs={
                               "language": "chinese"})
        except Exception as e:
            print(f"[ASRWorker] Error processing {audio_path}: {e}")
            return []

        # Transformers 的输出格式是 {'text': '...', 'chunks': [{'timestamp': (0.0, 2.0), 'text': '...'}]}
        # 将其转换为 OpenAI 格式: [{'start': 0.0, 'end': 2.0, 'text': '...'}]

        hf_chunks = result.get('chunks', [])
        openai_format_segments = []

        for chunk in hf_chunks:
            timestamp = chunk.get('timestamp')
            # 有时候 timestamp 可能是 None 或者 (None, None)
            if not timestamp:
                continue

            start, end = timestamp
            # 如果 end 是 None (最后一句话可能发生)，就赋值为 start + 2s (估算)
            if end is None:
                end = start + 2.0

            openai_format_segments.append({
                "start": start,
                "end": end,
                "text": chunk['text'].strip()
            })

        return openai_format_segments
