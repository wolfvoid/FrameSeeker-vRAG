<h1 align="center">FrameSeeker: A Multimodal RAG System for Long Videos</h1>

<p align="center">
    <a href="#">
        <img alt="Status" src="https://img.shields.io/badge/Status-Base_Version_MVP-orange">
    </a>
    <a href="#">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
    </a>
    <a href="#">
        <img alt="Framework" src="https://img.shields.io/badge/vLLM-0.14.1-green">
    </a>
    <a href="#">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

> ⚠️ **IMPORTANT NOTICE**
> 
> The current open-source code is a **Base Version (MVP)**. This version is designed to provide an **end-to-end runnable** baseline to demonstrate the feasibility of the entire long-video multimodal RAG pipeline. 
>
> **It has NOT yet undergone any engineering acceleration, VRAM optimization, or high-concurrency handling.** > 
> If you plan to reference this code or use it in a production environment, please be aware of the current performance bottlenecks. Deeply optimized versions targeting multi-GPU parallelism, retrieval latency, and VRAM footprint are currently under development (**Optimizations Coming Soon!**).

## 📖 Introduction

**FrameSeeker** is a Multimodal Retrieval-Augmented Generation (RAG) system tailored for long videos. It enables accurate content grounding, Q&A, and summarization of lengthy video files through multimodal feature extraction (visual frame extraction, ASR transcription), hybrid indexing (dense vectors + sparse keywords), and the generation capabilities of advanced MLLMs.

## 🚀 Model Zoo

FrameSeeker relies on the following open-source models to build its multimodal understanding and retrieval pipeline:

| Module Category | Model Name |
| :--- | :--- |
| **Visual Encoder** | `google/siglip2-base-patch16-224` |
| **Speech Recognition (ASR)** | `iic/Whisper-large-v3-turbo` |
| **Vector Embedding** | `BAAI/bge-m3` |
| **Reranker** | `BAAI/bge-reranker-v2-m3` |
| **Multimodal LLM (MLLM)** | `Qwen/Qwen3-VL-8B-Instruct` |

> 📌 **Best Practice / Warning**: When configuring models, **the Reranker and the Embedding model MUST belong to the same model family** (e.g., both from the `BAAI/bge` series). Mixing models from different families will result in mismatched feature spaces and cause a catastrophic drop in retrieval performance.

## 📂 Project Structure

```text
Video-RAG/
├── data/                        # Data storage directory
│   ├── raw_videos/              # Raw long video files (.mp4)
│   ├── processed_frames/        # Extracted keyframe images
│   └── vector_db/               # Vector database persistence files (ChromaDB/Milvus)
├── models/                      # Model download directory (symlink to shared storage recommended)
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Global configurations (GPU allocation, model paths)
│   ├── preprocessing/           # 1. Preprocessing module (Core for multi-GPU parallelism)
│   │   ├── video_loader.py      # Video frame extraction & audio separation
│   │   ├── asr_worker.py        # Whisper transcription (ASR)
│   │   ├── visual_encoder.py    # SigLIP visual feature extraction
│   │   └── pipeline.py          # Preprocessing pipeline scheduler (designed for multi-GPU)
│   ├── indexing/                # 2. Indexing module
│   │   ├── vector_store.py      # Vector DB CRUD operations (ChromaDB wrapper)
│   │   └── keyword_store.py     # BM25 inverted index construction
│   ├── retrieval/               # 3. Retrieval module (Search core)
│   │   ├── hybrid_searcher.py   # Hybrid recall logic (Dense + Sparse)
│   │   └── reranker.py          # BGE reranking logic
│   └── generation/              # 4. Generation module
│       └── vllm_engine.py       # vLLM wrapper for Qwen3-VL (TP parallelism)
├── app.py                       # Gradio frontend entry point
├── requirements.txt             # Project dependencies
└── run_ingestion.py             # Offline DB construction script (Run once)
```



## 🛠️ Quick Start
### 1. Environment Setup
We recommend using conda to create a clean Python environment:
```Bash
# 1. Create and activate a virtual environment
conda create -n vrag python=3.11 -y
conda activate vrag

# 2. Install a newer CUDA toolkit environment
conda install -c nvidia cuda-nvcc -y

# 3. Install vLLM (Specify version for compatibility)
pip install vllm==0.14.1

# 4. Install Flash Attention
# ⚠️ WARNING: This build process from source may take up to 1 hour or more. Please be patient.
MAX_JOBS=4 pip install flash_attn --no-build-isolation 

# 5. Install the remaining dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
Run the modules in the following order. Make sure you have downloaded the required models into the models/ directory and updated the paths in src/config.py.

Step 2.1: Video Processing & Vector Database Construction
```Bash
# Offline video frame extraction, feature encoding, and ASR transcription
python -m src.preprocessing.pipeline

# Build the vector database and BM25 index
python -m src.indexing.vector_store
```

Step 2.2: Start Retrieval & Generation Engines
```Bash
# Start the retrieval backend
python -m src.retrieval.engine

# Start the MLLM backend service (vLLM)
bash mllm.sh
python -m src.generation.generator
```

Step 2.3: Launch the Web UI (Gradio App)
```Bash
# Launch the frontend interaction interface (Binds to GPU 0 by default)
CUDA_VISIBLE_DEVICES=0 python app.py
```

## 🗓️ Optimization Roadmap
As mentioned at the beginning, this is a feasibility validation version. We are actively working on the following optimizations (Coming Soon):

- [x] v0.1: Base end-to-end pipeline runnable (MVP).
- [ ] v0.2: Preprocessing pipeline multi-processing / multi-GPU optimization.
- [ ] v0.3: Retrieval latency optimization (Milvus integration & async retrieval).
- [ ] v0.4: vLLM-based high-concurrency request optimization and VRAM defragmentation.
- [ ] v1.0: Complete one-click deployment Docker image and detailed benchmark reports.

## 🤝 Contributing
If you encounter bugs while running the base version, feel free to submit an Issue. Feedback regarding performance/speed will be addressed in upcoming optimization releases.

## 📜 License
This project is licensed under the MIT License. Third-party open-source models utilized in this project are subject to their respective original licenses.
