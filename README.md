# Emergency Department Admission Prediction System

A comprehensive machine learning and LLM-based system for predicting patient admission decisions in emergency departments using a multi-model fusion approach with LangGraph workflow orchestration.

## Project Overview

This project implements a hybrid decision-making system that combines:
- **Traditional ML Model**: Gradient Boosting Classifier for admission probability prediction
- **LLM Classifier**: Fine-tuned OpenBioLLM model for clinical text classification
- **Fusion Agent**: LLM-based reasoning agent that synthesizes multiple signals
- **LangGraph Workflow**: State-based workflow orchestration with human-in-the-loop capabilities

## Project Structure

```
Capstone_Organized/
├── 1-Data/                          # Training datasets and simulation database
├── 2-PreWorkflow_Dataset_Prep/      # Data preprocessing and blending notebooks
├── 3-Model_Training/                 # Model training pipelines
│   ├── 3.1-Traditional_ML/          # Traditional ML model training
│   └── 3.2-LLM_Classification/      # LLM fine-tuning and classification
├── 4-LangGraph/                      # LangGraph workflow implementation
└── 5-Evaluation_Reports/            # Model performance evaluation results
```

## Key Features

- **Multi-Model Fusion**: Combines ML and LLM predictions with clinical reasoning
- **Severity Gate**: Early detection of critical cases requiring immediate admission
- **Confidence-Based Routing**: Automatic routing to human review for low-confidence cases
- **Human-in-the-Loop (HITL)**: Manual override capability for edge cases
- **Comprehensive Logging**: Execution tracking and performance monitoring
- **Optimal Threshold Finding**: Data-driven threshold optimization for admission decisions

## Technologies Used

- **Machine Learning**: scikit-learn, XGBoost
- **LLM**: Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)
- **Workflow Orchestration**: LangGraph, LangChain
- **Data Processing**: pandas, numpy
- **Model Evaluation**: scikit-learn metrics

## Setup

### Prerequisites

```bash
pip install langgraph transformers peft accelerate bitsandbytes scikit-learn joblib pandas langchain-core langchain
```

### Model Artifacts

The system requires pre-trained model artifacts:
- ML model: `gb_model.joblib`
- ML preprocessor: `ml_preprocessor.joblib`
- Fine-tuned LLM: `OpenBioLLM_Final/`
- Feature columns: `ml_feature_columns.json`

## Usage

### Running the LangGraph Workflow

```python
from langgraph.graph import StateGraph

# Initialize workflow
inputs = {
    "visit_id": 1,
    "human_prompt": "Patient is 70yo, frail, and on chemotherapy."
}

config = {"configurable": {"thread_id": "sim-1"}}
final_state = graph.invoke(inputs, config)
```

### Evaluation

The system includes comprehensive evaluation notebooks that:
- Test on held-out test sets
- Calculate ROC-AUC scores
- Find optimal admission thresholds
- Generate performance reports

## Results

- **Agent AUC**: 0.7586
- **Optimal Threshold**: 0.3534 (Youden's J statistic)
- **Test Set Performance**: 73% accuracy with balanced precision/recall

## Data Privacy

The system implements PII redaction for:
- Email addresses
- Phone numbers
- SSNs
- Dates
- Age information
- Gender-specific terms

## License

[Add your license here]

## Authors

[Add author information here]

## Acknowledgments

[Add acknowledgments here]

