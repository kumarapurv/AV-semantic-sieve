# **AV Semantic Sieve: Intelligent Edge Data Uplift for L4 Autonomy**

### **Inspired by latest challenges in AI Pipelines for AV 2.0**

This project implements a **"Smart Sieve"** architecture designed to solve the "Data Tsunami" problem in Autonomous Driving. Instead of uploading every gigabyte of nominal driving data, this pipeline utilizes **Edge Uncertainty Quantification** and **Vision-Language Models (VLMs)** to identify and justify the "uplift" of high-signal long-tail edge cases.

## **🚀 The Core Problem: 99.9% of Data is Boring**

At commercial scale, an L4 fleet generates petabytes of data that are economically impossible to store. This project demonstrates an **Intelligent Filtering** approach:

1. **Tier 1 (Perception Sieve):** Uses a lightweight detector (YOLOv8) to scan for high-entropy frames (where the model is "confused").  
2. **Tier 2 (Semantic Guardian):** Employs a local, quantized VLM (Phi-3.5 Vision) to act as a Safety Auditor, generating natural language justifications for data retention.

## **🛠 Tech Stack**

* **Hardware:** Optimized for Apple Silicon (M2 Pro) using mlx-vlm.  
* **Dataset:** nuImages (2D Long-Tail subset of nuScenes).  
* **Inference:** 4-bit Quantized Phi-3.5 Vision & YOLOv8-Nano.  
* **Acceleration:** MPS (Metal Performance Shaders) for GPU acceleration.

## **📁 Project Structure**

.  
├── data/               \# nuImages dataset (ignored by git)  
├── src/  
│   └── unified\_pipeline.py  \# The main "Smart Sieve" executable  
├── portfolio\_results/  \# Generated Audit reports and sieve visualizations  
├── .gitignore          \# Data & model exclusion rules  
└── requirements.txt    \# Reproducible environment dependencies

## **📊 Strategic Outcome**

* **OpEx Reduction:** Simulates a 90%+ reduction in bandwidth and cloud storage costs.  
* **SOTIF Compliance:** Implements an independent "Guardian" monitor to prevent common-mode failures in the primary perception stack.  
* **Closed-Loop Active Learning:** Feeds high-entropy edge cases back into the labeling pipeline to strengthen the model's decision boundaries.
