# Integrating Generative AI and Agentic Reasoning for Predictive Decision Support in Emergency Departments

### University of Chicago · MS Applied Data Science Research Capstone

This repository contains the research artifacts, experiments, and evaluation notebooks for our study on **agentic AI reasoning for real-time Emergency Department (ED) admission prediction**.  
Our work explores how **traditional ML models**, **fine-tuned biomedical LLMs**, and **human-in-the-loop (HITL) feedback** can be integrated via an **agent-based architecture** to improve early triage decision-making.

---

## Abstract

Accurate early prediction of Emergency Department (ED) admissions is crucial for effective resource planning, patient safety, and efficient clinical workflow.  
However, traditional models rely primarily on structured data (e.g., vital signs) or unstructured data, missing key signal combinations in both approaches.  
Meanwhile, modern LLMs provide strong text-understanding capabilities but lack calibration, traceability, and clinical safety controls.

We propose an **Agentic ED Admission AI System** that fuses:

- **Structured ML predictions (XGBoost)**  
- **Fine-tuned biomedical LLM sequence classifier (OpenBioLLM-8B)**  
- **Real-time human input**  
- **Agentic orchestration via LangGraph**

Our results show that agentic reasoning significantly improves recall, F1 score, and clinical interpretability compared to standalone ML or LLM models.

---

## Repository Contents

```
├── 1-Data/                      # Synthetic ED dataset (de-identified)
├── 2-PreWorkflow_Dataset_Prep/  # Notebooks for data cleaning & integration
├── 3-Model_Training/            # ML and LLM training artifacts (not all uploaded)
│   ├── 3.1-Traditional_ML/      # Gradient Boosting classifier
│   └── 3.2-LLM_Classification/  # Fine-tuned OpenBioLLM notebooks
├── 4-LangGraph/                 # Agentic workflow notebooks
│   ├── 4.0-LangGraph_Logs/      # Execution traces
│   └── 4.1-LangGraph_Agent_with_Reasoning.ipynb
├── 5-Evaluation_Reports/        # Confusion matrices, metrics, plots
└── README.md                    # (this file)
```

---

## Methodology

### 1. Multisource Inputs
Our system integrates four distinct information sources:

| Modality | Example Features | Strength |
|---------|------------------|----------|
| **Structured vitals** | HR, BP, RR, temp, SpO₂ | High signal for physiological severity |
| **Unstructured triage notes** | Free-text written by nurses | Rich contextual clinical reasoning |
| **ESI score** | 1–5 emergency severity scale | Clinically validated triage heuristic |
| **Human note (optional)** | Real-time clinician context | Overwrite or augment model reasoning |

---

## 2. Agentic Architecture

A multi-stage agent graph is constructed using **LangGraph**, enabling controlled interaction between ML, LLM, and human agents.

![workflow](https://ik.imagekit.io/monicako/graph.png)

### Workflow Summary

1. **Severity Gate** - custimized clinical rules trigger auto-admission.  
2. **Parallel Model Execution** — ML classifier, LLM classifier, and human context operate simultaneously.  
3. **Fusion Module** — integrates outputs into a unified probability + rationale.  
4. **Confidence Check** — low-confidence predictions escalate to human review.  
5. **Finalization** — outputs decision, probability, rationale, and trace log.  

---

## Experimental Setup

- **Dataset size:** 4,200 synthetic ED visits  
- **Train/test split:** 80/20 stratified  
- **Models used:** Gradient Boosting + fine-tuned OpenBioLLM-8B  
- **Metrics evaluated:** Precision, Recall, F1, AUC, confusion matrices  

---

## Results

### Baseline vs Agent Models

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Gradient Boosting | 0.35 | 0.11 | 0.17 |
| Fine-tuned OpenBioLLM | 0.34 | 0.32 | 0.33 |
| **Agent (No Reasoning)** | 0.58 | 0.52 | 0.55 |
| **Agent (Reasoning Enabled)** | **0.55** | **0.64** | **0.59** |

### Performance Gains
- **+26% Admit F1 score**  
- **+8% accuracy**  
- **+21% AUC**  

---

## Key Contributions

- **Multimodal clinical fusion**  
- **Agentic reasoning framework**  
- **Human-in-the-loop override and context injection**  
- **Improved sensitivity and fairness**  
- **Transparent, auditable decision traces**  

---

## Usage

Since the repository is currently notebook-driven, usage follows:

```
2-PreWorkflow_Dataset_Prep/        → data preparation
3-Model_Training/                  → ML & LLM training
4-LangGraph/                       → run agentic workflow
5-Evaluation_Reports/              → view metrics
```

---

## Large Files

Model checkpoints (e.g., OpenBioLLM finetuned weights) and databases exceed GitHub limits and are not included.

---

## Contributors

- Cassandra Chen  
- Monica Ko  
- Jane Lee  
- Alvin Yao  
- Faculty Advisor: Dr. Utku Pamuksuz 
