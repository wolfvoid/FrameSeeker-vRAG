# FrameSeeker
a Multimodal Rag System for long videos

Video-RAG/
├── data/                        # 数据存放目录
│   ├── raw_videos/              # 原始长视频文件 (.mp4)
│   ├── processed_frames/        # 抽取的关键帧图片
│   └── vector_db/               # 向量数据库持久化文件 (ChromaDB/Milvus)
├── models/                      # 模型下载目录 (建议软链接到共享存储)
├── src/                         # 源代码
│   ├── __init__.py
│   ├── config.py                # 全局配置 (GPU分配、模型路径)
│   ├── preprocessing/           # 1. 预处理模块 (多卡并行核心)
│   │   ├── video_loader.py      # 视频抽帧 & 音频分离
│   │   ├── asr_worker.py        # Whisper 转录 (ASR)
│   │   ├── visual_encoder.py    # SigLIP 视觉特征提取
│   │   └── pipeline.py          # 调度 10 张卡的预处理流水线
│   ├── indexing/                # 2. 索引模块
│   │   ├── vector_store.py      # 向量库增删改查 (ChromaDB 封装)
│   │   └── keyword_store.py     # BM25 倒排索引构建
│   ├── retrieval/               # 3. 检索模块 (搜索核心)
│   │   ├── hybrid_searcher.py   # 混合召回逻辑 (Dense + Sparse)
│   │   └── reranker.py          # BGE 重排逻辑
│   └── generation/              # 4. 生成模块
│   │   └── vllm_engine.py       # Qwen2.5-VL 的 vLLM 封装 (TP并行)
├── app.py                       # Gradio 前端入口
├── requirements.txt             # 依赖包
└── run_ingestion.py             # 离线建库脚本 (一次性运行)


model_list =[
    "google/siglip2-base-patch16-224",
    "iic/Whisper-large-v3-turbo",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-VL-8B-Instruct"
]
reranker和embedding需要用同一个模型家族的模型，否则效果会很差


conda create -n vrag python=3.11 -y
conda activate vrag
conda install -c nvidia cuda-nvcc # 安装较新版的cuda环境

pip install vllm==0.14.1
MAX_JOBS=4 pip install flash_attn --no-build-isolation (this may takes time, up to 1 hour or even more)
pip install -r requirements.txt

python -m src.preprocessing.pipeline
python -m src.indexing.vector_store
python -m src.retrieval.engine

bash mllm.sh
python -m src.generation.generator
CUDA_VISIBLE_DEVICES=0 python app.py
