import os
import json
import torch
import multiprocessing
from tqdm import tqdm

from src.config import Config
from src.preprocessing.video_loader import VideoLoader
from src.preprocessing.asr_worker import ASRWorker
from src.preprocessing.visual_encoder import VisualEncoder


def process_single_video(args):
    """
    单个视频的处理逻辑(单GPU)
    args: (video_path, gpu_id)
    """
    video_path, gpu_id = args
    device = f"cuda:{gpu_id}"
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_json_path = os.path.join(Config.PROCESSED_DIR, f"{video_id}.json")

    # 如果已经处理过，跳过
    if os.path.exists(output_json_path):
        print(f"[Pipeline] {video_id} already processed, skipping.")
        return

    print(f"--- GPU {gpu_id} processing: {video_id} ---")

    # 1. 抽帧 & 抽音频
    print(f"[Pipeline] Step1 >>> Extracting frames and audio for {video_id} on GPU {gpu_id}...")
    loader = VideoLoader(frame_interval=Config.FRAME_INTERVAL)  # 提取关键帧
    frames_info = loader.extract_frames(video_path, Config.FRAMES_DIR)
    audio_path = os.path.join(Config.FRAMES_DIR, f"{video_id}.mp3")  # 提取音频
    loader.extract_audio(video_path, audio_path)

    # 2. ASR 转录 (加载 Whisper 到显存)
    print(f"[Pipeline] step2 >>> Transcribing audio for {video_id} on GPU {gpu_id}...")
    asr_worker = ASRWorker(Config.MODEL_PATH['audio'], device)
    transcript = asr_worker.transcribe(audio_path)
    del asr_worker  # 释放显存
    torch.cuda.empty_cache()

    # 3. 视觉特征提取 (加载 SigLIP 到显存)
    print(f"[Pipeline] step3 >>> Encoding visual features for {video_id} on GPU {gpu_id}...")
    vis_worker = VisualEncoder(Config.MODEL_PATH['visual'], device)
    image_paths = [f['frame_path'] for f in frames_info]
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i+batch_size]
        embs = vis_worker.encode_images(batch_paths)  # numpy array
        all_embeddings.append(embs)
    del vis_worker  # 释放显存
    torch.cuda.empty_cache()

    import numpy as np
    visual_embeddings = np.vstack(all_embeddings)
    # 4. 数据对齐与保存
    # 把 帧信息(Visual) 和 字幕信息(Audio) 合并存起来
    print(f"[Pipeline] step4 >>> Saving processed data for {video_id} on GPU {gpu_id}...")
    final_data = {
        "video_id": video_id,
        "frames": [],     # 存视觉向量
        "transcript": []  # 存文本
    }
    # 存帧数据
    for idx, info in enumerate(frames_info):
        info['embedding'] = visual_embeddings[idx].tolist()  # 转 list 存 json
        final_data['frames'].append(info)
    # 存字幕数据
    final_data['transcript'] = transcript
    # 写入文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False)

    print(f"--- GPU {gpu_id} finished: {video_id} ---")


def run_parallel_ingestion():
    """多进程并行调度入口"""
    # 扫描所有视频
    video_files = [os.path.join(Config.VIDEO_DIR, f) for f in os.listdir(
        Config.VIDEO_DIR) if f.endswith('.mp4')]

    if not video_files:
        print("No videos found in data/raw_videos!")
        return

    # 准备任务分配：[(video1, gpu0), (video2, gpu1), ...]
    tasks = []
    available_gpus = Config.PREPROCESS_GPUS
    num_gpus = len(available_gpus)

    for i, video_path in enumerate(video_files):
        assigned_gpu = available_gpus[i % num_gpus]  # 轮询分配
        tasks.append((video_path, assigned_gpu))

    print(
        f"Starting parallel processing on {num_gpus} GPUs for {len(tasks)} videos...")

    # 启动多进程
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        list(tqdm(pool.imap_unordered(process_single_video, tasks), total=len(tasks)))


if __name__ == "__main__":
    run_parallel_ingestion()
