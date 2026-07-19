# GeoQuery Workspace 🌍

GeoQuery Workspace is an advanced, multimodal geospatial intelligence platform designed to automate object detection, structural footprint analysis, and natural language Q&A from high-resolution satellite and aerial imagery. 

By combining cutting-edge computer vision slicing pipelines with a localized Large Language Model (LLM), GeoQuery transforms raw pixels into structured, searchable spatial intelligence—running completely on local hardware acceleration.

---

## 🚀 Core Architecture & Features

- **Hyper-Resolution Sliced Inference:** Powered by the **SAHI** framework and a custom-trained **YOLOv8** model (30 epochs), the pipeline segments large aerial imagery into non-overlapping grid matrices to capture microscopic infrastructure details without data loss.
- **Dynamic Footprint Classification:** Automatically extracts object coordinate parameters to dynamically scale structural objects into *Small*, *Medium*, or *Large* footprint attributes.
- **Localized Multimodal QA Engine:** Leverages `Qwen2.5-1.5B-Instruct` locally to synthesize rich scene captions and handle complex, text-based spatial reasoning via an isolated context window.
- **Hardware-Optimized Memory Gatekeeping:** Explicitly written to utilize PyTorch's Apple Silicon (`mps`) backend with aggressive VRAM cache clearing and state preservation to prevent memory leaks during high-volume batch processing.

---

## 🛠️ Project Structure

```text
geoquery/
├── stream.py            # Main Streamlit application and execution lifecycle
├── .gitignore           # Active environment and OS junk exclusion file
├── requirements.txt     # Complete hardware and library dependencies
└── README.md            # System documentation
