# StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding

## 📦 1. Model and Data Preparation

This study builds upon the M3-Agent framework; therefore, the M3-Agent model and the corresponding datasets must be downloaded and prepared in advance.

### 1.1. Model: M3-Agent

- **Description**: The first streaming video model based on Agent Memory.
- **Access**: [ByteDance-Seed/M3-Agent](https://github.com/ByteDance-Seed/m3-agent)  
  📄 *Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory*, Arxiv 2025.

### 1.2. Data

- **Description**: The memory graphs of M3-Bench-robot and M3-Bench-web.
- **Access**: [ByteDance-Seed/M3-Agent](https://github.com/ByteDance-Seed/m3-agent)  
  📄 *Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory*, Arxiv 2025.

For detailed information on the required resources and downloading procedures, please refer to the links provided by M3-Agent. The overall structure of this project is organized as follows:

```text
StreamMeCo-main/

├── configs/
│   ├── __init__.py
│   ├── api_config.json
│   ├── memory_config.json
│   └── processing_config.json
│
├── data/
│   ├── annotations/
│   │   ├── robot.json
│   │   ├── web.json
│   │   └── videomme.json
│   │
│   ├── videos/
│   │   ├── robot/
│   │   │   └── ...
│   │   ├── web/
│   │   │   └── ...
│   │   └── videomme/
│   │       └── ...
│   │
│   └── memory_graphs/
│       ├── robot/
│       │   └── ...
│       ├── web/
│       │   └── ...
│       └── videomme/
│           └── ...
│
├── m3_agent/
│   ├── control.py
│   ├── memorization_intermediate_outputs.py
│   └── memorization_memory_graphs.py
│
├── mmagent/
│   ├── src/
│   │   ├── face_clustering.py
│   │   └── face_extraction.py
│   │
│   ├── utils/
│   │   ├── chat_api.py
│   │   ├── chat_qwen.py
│   │   ├── general.py
│   │   ├── video_processing.py
│   │   └── video_verification.py
│   │
│   ├── __init__.py
│   ├── face_processing.py
│   ├── memory_processing.py
│   ├── memory_processing_qwen.py
│   ├── prompts.py
│   ├── retrieve.py
│   ├── videograph.py
│   └── voice_processing.py
│
├── models/
│   ├── M3-Agent-Control/
│   │   └── ...
│   ├── M3-Agent-Memorization/
│   │   └── ...
│   └── pretrained_eres2netv2.ckpt
│
├── speakerlab/
│   └── ...
│
├── cut_videomme.py
├── memory_videomme.jsonl
├── requirements.txt
├── score.py
├── setup.sh
├── streammeco.py
└── visualization.py                                   
```

---

## ⚙️ 2. Environment Setup

We recommend using a Python virtual environment to avoid conflicts.

### 2.1. Create and activate a virtual environment (e.g., with conda):

```bash
conda create -n streammeco python=3.11.14 -y
conda activate streammeco
```

### 2.2. Install Python dependencies:

```bash
bash setup.sh
pip install qwen-omni-utils==0.0.4
pip install transformers==4.51.0
pip install vllm==0.8.4
pip install numpy==1.26.4
```

---

## 🛠️ 3. Memory Graph Generation

For the M3-Bench-robot and M3-Bench-web datasets, the memory graphs provided by the official M3-Agent implementation can be directly used. In contrast, for the VideoMME dataset, the memory graphs need to be constructed from scratch, and the detailed procedure is described as follows.

### 3.1. Cut Video

You need to split each video into 30-second segments. This can be done using the script `cut_videomme.py`.

### 3.2. Prepare the JSONL files

You need to prepare a JSONL file to specify the storage paths of video segments, memory graphs, and intermediate outputs. We provide such a file named `memory_videomme.jsonl`.

### 3.3. Generate Intermediate Outputs

This step uses Face Detection and Speaker Diarization tools to generate intermediate outputs. You can run the following code.

```bash
python -m m3_agent.memorization_intermediate_outputs --data_file /StreamMeCo-main/memory_videomme.jsonl
```

### 3.4. Generate Memory Graphs

This step uses the M3-Agent-Memorization model to generate memory graphs. You can run the following code.

```bash
python -m m3_agent.memorization_memory_graphs --data_file /StreamMeCo-main/memory_videomme.jsonl
```

---

## ✅ 4. Memory Graph Compression

You can use our **StreamMeCo** framework to compress the previously generated memory graphs, as detailed below.

```bash
python streammeco.py
```

You can control the dataset selection and the saving path of the compressed memory graphs by specifying the `--mem_path` and `--compressed_mem_path` parameters.

---

## 🚀 5. Inference

You can run the model using the compressed memory graphs with the following code.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m m3_agent.control
```
Note that you should modify the `--data_file` parameter and replace it with the path to the compressed memory graphs.

---

## 🔍 6. Memory Graph Visualization

If you need to inspect the contents of the memory graphs, please run the following code.

```bash
python visualization.py
```

Note that the memory graph path specified by `--mem_path` should be replaced with the one you intend to inspect.

---

## 🙏 7. Acknowledgements

- We sincerely thank the developers of the [**ByteDance-Seed/M3-Agent**](https://github.com/ByteDance-Seed/m3-agent) model for their outstanding work and for making their codebase publicly available.

---

## 📬 8. Contact

If you have any questions or encounter any issues, feel free to open an issue or contact me directly.







