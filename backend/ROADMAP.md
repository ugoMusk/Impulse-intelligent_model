# Impulse Intelligent Model (IIMo) — Development Roadmap

This document defines a **complete, implementation-ready roadmap** for building the **Impulse Intelligent Model (IIMo)**.

It aligns with:
- Hybrid AI Architecture (Transformer + RAG + Memory + Edge AI)
- TensorFlow-based model training
- Modular AI system design
- FastAPI inference layer
- Edge Impulse integration (hackathon requirement)

---

## 0. Core System Principles

### 0.1 Separation of Concerns

| Component            | Responsibility |
|---------------------|---------------|
| Transformer Model   | Text generation & reasoning |
| Retrieval Module    | External knowledge access |
| Memory Module       | Context persistence |
| Edge Module         | Real-time edge inference (Edge Impulse) |
| Inference Engine    | Pipeline orchestration |
| API Layer           | External communication |

---

### 0.2 Training vs Inference Boundary

- ❗ The model is **NOT trained on live retrieval or edge data**
- ❗ Retrieval, Memory, and Edge Impulse are used **ONLY during inference**

```
TRAINING:  Instruction + Context → Output  
INFERENCE: Query → Routing → Retrieval + Memory + Edge → Model
```

---

### 0.3 Standard Input Format (CRITICAL)

All model interactions MUST follow:

```
Instruction: <task>
Context: <retrieved knowledge>
Memory: <past context>
Edge Signals: <optional>
Output:
```

---

## 1. Project Structure

```
Impulse-intelligent_model/
├── backend/
│   ├── api/
│   │   ├── server.py
│   │   └── routes.py
│   │
│   ├── inference/
│   │   ├── pipeline.py
│   │   ├── router.py              # Input Router (NEW)
│   │   ├── prompt_builder.py
│   │   └── output_formatter.py
│   │
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── retrieve.py
│   │   └── index_data.py
│   │
│   ├── memory/    
│   │   ├── memory_store.py
│   │   ├── memory_retriever.py
│   │   └── memory_updater.py
│   │
│   ├── edge_impulse/                     
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── client.py
│   │   ├── ingestion.py
│   │   ├── training.py
│   │   ├── device_manager.py
│   │   ├── pipeline.py
│   │   └── retry_queue.py
│   │
│   ├── model/
│   │   ├── transformer.py
│   │   ├── tokenizer.py
│   │   └── model_loader.py
│   │
│   ├── training/
│   │   ├── train_model.py
│   │   └── data_loader.py
│   │
│   ├── monitoring/
|   |    ├── metrics.py
|   |    ├── logger.py
|   |    ├── tracer.py
|   |    ├── feedback.py
|   |    └── evaluator.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── configs/
│   ├── model_config.yaml
│   └── edge_config.yaml
│
├── scripts/
├── docs/
└── requirements.txt
```

---

## 2. Environment Setup

### 2.1 Install Dependencies

```bash
pip install tensorflow fastapi uvicorn qdrant-client numpy pandas sentencepiece requests
```

Optional:
```bash
pip install pytest black isort flake8 mypy
```

---

### 2.2 Hardware Considerations

- GPU recommended (training)
- CPU acceptable for inference (TFLite)
- Edge Impulse runs externally (API-based)

---

## 3. Data Pipeline

### 3.1 Dataset Collection

| Dataset Type | Purpose |
|-------------|--------|
| Coding | Code generation |
| Reasoning | Multi-step logic |
| QA | Knowledge grounding |

---

### 3.2 Data Formatting

```
Instruction: <task>
Context: <optional>
Output: <expected result>
```

---

### 3.3 Tokenization

- Use SentencePiece
- Save:
```
backend/model/tokenizer.model
```

---

### 3.4 Data Loader

File:
```
backend/training/data_loader.py
```

Responsibilities:
- Load datasets
- Format data
- Tokenize
- Build `tf.data.Dataset`

---

## 4. Model Development (TensorFlow)

### 4.1 Transformer Implementation

File:
```
backend/model/transformer.py
```

---

### 4.2 Model Interface

```python
class IIMoModel(tf.keras.Model):
    def call(self, input_ids, attention_mask):
        ...
```

---

### 4.3 Configuration

```
configs/model_config.yaml
```

---

## 5. Training Pipeline

File:
```
backend/training/train_model.py
```

Steps:
1. Load data  
2. Tokenize  
3. Build model  
4. Train  
5. Save  

---

## 6. Retrieval Module (RAG)

### Files:
```
backend/retrieval/embedder.py
backend/retrieval/retrieve.py
backend/retrieval/index_data.py
```

---

## 7. Memory Module (MVP FEATURE)

### Files:
```
backend/memory/memory_store.py
backend/memory/memory_retriever.py
backend/memory/memory_updater.py
```

### Responsibilities:
- Store interactions
- Retrieve contextual history
- Update memory after inference

---

## 8. Edge AI Module (Edge Impulse Integration)

### Files:
```
backend/edge/edge_client.py
backend/edge/edge_inference.py
backend/edge/edge_router.py
backend/edge/edge_config.py
```

---

### 8.1 Integration Rules (CRITICAL)

- Edge Impulse is used **ONLY for edge inference**
- NEVER replaces transformer model
- Used for:
  - Audio classification
  - Sensor data processing
  - Real-time signals

---

### 8.2 Edge Flow

```
Input → Edge Router → Edge Impulse API → Result → Pipeline
```

---

## 9. Inference Engine

### File:
```
backend/inference/pipeline.py
```

---

### 9.1 Full Pipeline

```
User Input
   ↓
Input Router
   ↓
 ┌──────────────┬──────────────┬──────────────┐
 │ Retrieval    │ Memory       │ Edge         │
 │ (Qdrant)     │              │ (Impulse)    │
 └──────┬───────┴──────┬───────┴──────┬───────┘
        ↓              ↓              ↓
        Combined Context + Signals
                     ↓
              Prompt Builder
                     ↓
              Tokenization
                     ↓
              Transformer Model
                     ↓
              Output Formatter
                     ↓
              Memory Update
                     ↓
                Response
```

---

## 10. API Layer (FastAPI)

### Files:
```
backend/api/server.py
backend/api/routes.py
```

---

### Endpoint:

```python
@app.post("/generate")
```

---

## 11. Evaluation

File:
```
backend/evaluation/evaluate.py
```

---

## 12. Testing

- Unit tests
- Pipeline tests
- Edge integration tests (IMPORTANT)

---

## 13. Deployment

```bash
uvicorn backend.api.server:app --reload
```

---

## 14. Milestones

| Phase | Deliverable |
|------|------------|
| 1 | Setup |
| 2 | Data |
| 3 | Model |
| 4 | Training |
| 5 | Retrieval |
| 6 | Memory |
| 7 | Edge Integration |
| 8 | Inference |
| 9 | API |
| 10 | Evaluation |
| 11 | Deployment |

---

## Final Summary

This roadmap ensures:

- Proper modular architecture  
- Correct RAG implementation  
- Integrated memory system (MVP)  
- Edge Impulse compliance (hackathon-ready)  
- Production-ready TensorFlow pipeline  

The system is now:

- Fully specified  
- Architecturally consistent  
- Ready for implementation without ambiguity  

**IIMo is now a complete hybrid AI system blueprint.**