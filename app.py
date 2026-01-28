import os
import gradio as gr
from src.config import Config
from src.retrieval.engine import VideoSearchEngine
from src.generation.generator import VLLMGenerator

def create_ui(search_engine, generator):
    """
    UI 构建函数
    """
    def query_pipeline(video_name, user_query):
        """核心处理链路"""
        if not user_query:
            return "Please enter a question.", "No query provided.", 0

        try:
            # 1. 检索 (Retrieval) - 运行在本地 GPU 0
            print(f"Searching for: {user_query}")
            retrieval_results = search_engine.search(user_query, top_k=5)

            # 2. 生成 (Generation) - 发送请求给 localhost:8000
            print(f"Generating answer...")
            answer = generator.generate(user_query, retrieval_results)

            # 3. 格式化输出
            evidence_html = "<h4>📚 Retrieval Evidence:</h4>"
            timestamps = []

            for res in retrieval_results:
                score = res.get('score', 0)
                if res['type'] == 'visual':
                    evidence_html += f"""
                    <div style='margin-bottom: 8px; padding: 5px; background: #e6f3ff; border-radius: 4px;'>
                        <small><b>[Visual]</b> Score: {score:.3f} | Time: {res['timestamp']}s</small><br>
                        <img src='/file={res['path']}' style='height: 100px; object-fit: cover;'>
                    </div>
                    """
                    timestamps.append(res['timestamp'])
                else:
                    evidence_html += f"""
                    <div style='margin-bottom: 8px; padding: 5px; background: #f0f0f0; border-radius: 4px;'>
                        <small><b>[Audio]</b> Score: {score:.3f} | Time: {res['start']}-{res['end']}s</small><br>
                        <i>"{res['content']}"</i>
                    </div>
                    """
                    timestamps.append(res['start'])

            seek_time = timestamps[0] if timestamps else 0
            return answer, evidence_html, seek_time

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", "", 0

    # === 构建 UI 界面 ===
    with gr.Blocks(title="FrameSeeker vRAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎞️ FrameSeeker: 长视频细粒度多模态问答系统")

        with gr.Row():
            # 左侧：视频播放区
            with gr.Column(scale=2):
                default_video = None
                if os.path.exists(Config.VIDEO_DIR):
                    video_files = [f for f in os.listdir(
                        Config.VIDEO_DIR) if f.endswith('.mp4')]
                    if video_files:
                        default_video = os.path.join(
                            Config.VIDEO_DIR, video_files[0])

                video_player = gr.Video(
                    value=default_video,
                    label="Source Video",
                    height=400
                )
                seek_state = gr.State(value=0)

            # 右侧：对话区
            with gr.Column(scale=3):
                chatbot = gr.Markdown(
                    label="Answer", value="Waiting for question...")
                msg = gr.Textbox(
                    label="Ask a question about the video", placeholder="e.g., Why use ReLU?")
                submit_btn = gr.Button("Search & Answer", variant="primary")
                evidence_box = gr.HTML(label="Evidence")

        # 按钮点击事件
        submit_btn.click(
            fn=query_pipeline,
            inputs=[video_player, msg],
            outputs=[chatbot, evidence_box, seek_state]
        )

    return demo


if __name__ == "__main__":
    print(">>> Main process started.")

    # 1. 初始化 Search Engine
    print(">>> Initializing Search Engine (SigLIP/BGE) on cuda:0 ...")
    search_engine = VideoSearchEngine(device="cuda:0")

    # 2. 初始化 Generator (API Client)
    print(">>> Initializing API Client connecting to localhost:8000 ...")
    generator = VLLMGenerator(api_url="http://localhost:8000/v1")

    # 3. 启动 UI
    print(">>> Launching Gradio...")
    demo = create_ui(search_engine, generator)
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=True, allowed_paths=["/"])
