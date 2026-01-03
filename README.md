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
StreamMeCo/

├── configs/
│   ├── __init__.py
│   ├── api_config.json
│   ├── memory_config.json
│   └── processing_config.json
│
├── data/
│   ├── annotations/
│   │   ├── robot.json
│   │   ├── videomme.json
│   │   └── web.json
│   │
│   ├── datasets/
│   │   ├── M3-Bench-robot/
│   │   │   └── ...
│   │   │
│   │   ├── M3-Bench-videomme/
│   │   │   └── ...
│   │   │
│   │   └── M3-Bench-web/
│   │       └── ...
│   │
│   └── memory_graphs/
│       ├── robot/
│       │   └── ...
│       │
│       ├── web/
│       │   └── ...
│       │
│       └── videomme/
│           └── ...
|
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
|——models/
|     ├── M3-Agent-Control/
│     │     └── ...
│     │
│     ├── M3-Agent-Memorization/
│     │     └── ...
|
|——speakerlab/
|     └── ...
|
├── score.py
├── streammeco.py
├── visualization.py
├── setup.sh
└── requirements.txt

                                   
```

---

