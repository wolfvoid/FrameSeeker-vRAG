import cv2
import os
import subprocess
from tqdm import tqdm
from typing import List, Tuple


class VideoLoader:
    def __init__(self, frame_interval: int = 1):
        self.frame_interval = frame_interval

    def extract_audio(self, video_path: str, output_audio_path: str):
        """调用 ffmpeg 提取音频 (极速模式)"""
        if os.path.exists(output_audio_path):
            print(f"[VideoLoader] Audio already exists: {output_audio_path}")
            return
        # -y: 覆盖, -vn: 去除视频, -acodec: 音频编码
        print(f"[VideoLoader] Extracting audio for {os.path.basename(video_path)}...")
        cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-y", output_audio_path, "-loglevel", "error"]
        subprocess.run(cmd, check=True)

    def extract_frames(self, video_path: str, output_dir: str) -> List[dict]:
        """每隔N秒抽取一帧，返回帧信息列表"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_info = []
        step = int(fps * self.frame_interval)

        frame_idx = 0
        saved_count = 0

        pbar = tqdm(total=total_frames, unit='frame',
                    desc=f"Extracting {video_id}", mininterval=1.0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只保存关键时间点的帧
            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                frame_filename = f"{video_id}_{int(timestamp)}s.jpg"
                save_path = os.path.join(output_dir, frame_filename)

                cv2.imwrite(save_path, frame)
                frames_info.append({
                    "timestamp": timestamp,
                    "frame_path": save_path,
                    "frame_id": frame_idx
                })
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        print(
            f"[VideoLoader] Processed {video_path}: {saved_count} frames extracted.")
        return frames_info
